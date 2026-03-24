module LLaMa2.Layer.Attention.QKVProjection
  (
    qkvProjectionController
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( HeadDimension,
      NumKeyValueHeads,
      NumLayers,
      NumQueryHeads,
      RotaryPositionalEmbeddingDimension,
      SequenceLength )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.RmsNormSeq (rmsNormSeq)
import LLaMa2.Types.LayerData (ActivationBramAddr)
import qualified LLaMa2.Layer.Attention.FSM as FSM (processingControllerFSM)

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Arbiter as ARB
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified LLaMa2.Memory.FPVecLoader as FPVec
import qualified LLaMa2.Layer.Attention.QueryHeadProjector as QHP (queryHeadProjector, QHeadDebugInfo)
import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector as KVHP

--------------------------------------------------------------------------------
-- QKV projector
--------------------------------------------------------------------------------

-- | Coordinates all query heads and key-value heads for QKV projection.
--
-- == Overview
--
-- This component instantiates NumQueryHeads query projectors and NumKeyValueHeads
-- KV projectors, combining their outputs. It provides a single interface for
-- the complete QKV projection stage of the attention mechanism.
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────────────────┐
--                    │                    qkvProjector                         │
--                    │                                                         │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │              Query Heads (×NumQueryHeads)       │    │
--                    │  │                                                 │    │
--   inputValid ─────►│  │  ┌─────────┐ ┌─────────┐     ┌─────────┐        │    │
--                    │  │  │ QHead 0 │ │ QHead 1 │ ... │ QHead N │        │    │
--   downStreamReady─►│  │  │         │ │         │     │         │        │    │
--                    │  │  └────┬────┘ └────┬────┘     └────┬────┘        │    │
--   xVec ───────────►│  │       │           │               │             │    │
--                    │  │       ▼           ▼               ▼             │    │
--   seqPos ─────────►│  │   qVecs[0]    qVecs[1]  ...   qVecs[N]          │    │
--                    │  │   qValids[0]  qValids[1] ... qValids[N]         │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │              KV Heads (×NumKeyValueHeads)       │    │
--                    │  │                                                 │    │
--                    │  │  ┌─────────┐ ┌─────────┐     ┌─────────┐        │    │
--                    │  │  │ KVHead0 │ │ KVHead1 │ ... │ KVHeadM │        │    │
--                    │  │  │ (K & V) │ │ (K & V) │     │ (K & V) │        │    │
--                    │  │  └────┬────┘ └────┬────┘     └────┬────┘        │    │
--                    │  │       │           │               │             │    │
--                    │  │       ▼           ▼               ▼             │    │
--                    │  │   kVecs[0]    kVecs[1]  ...   kVecs[M]          │    │
--                    │  │   vVecs[0]    vVecs[1]  ...   vVecs[M]          │    │
--                    │  │   kvValids[0] kvValids[1]... kvValids[M]        │    │
--                    │  │                                                 │    │
--                    │  └────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    │  outputValid = AND(all qValids) AND AND(all kvValids)   │
--                    │  readyForInput = AND(all qReadys) AND AND(all kvReadys) │
--                    │                                                         │
--                    └─────────────────────────────────────────────────────────┘
-- @
--
-- == Coordination Strategy
--
-- All heads (Q, K, V) share a single AXI arbiter and use coordinated clearing:
--
-- @
-- consumeSignal = outputValid .&&. downStreamReady
-- @
--
-- Q heads pass @pure True@ as downStreamReady to the per-row FSM (always
-- accepts the next row); their output latch clears only via consumeSignal.
--
-- KV heads follow the same pattern: @pure True@ for the FSM, consumeSignal
-- for latch clearing. K and V within each KV head are independent compute
-- paths, both governed by the same consumeSignal.
--
-- The AXI arbiter routes Q + K + V masters (NumQueryHeads + 2*NumKeyValueHeads
-- total) to the single DRAM slave.
--
-- Slot 0 base address in the activation BRAM (inputVector slot).
slot0BramBase :: ActivationBramAddr
slot0BramBase = 0

qkvProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom FixedPoint             -- ^ bramRdData: BRAM output (slot 0, 1-cycle latency from bramRdAddr)
  -> ( Master.AxiMasterOut dom
     , Vec NumKeyValueHeads (Signal dom (Maybe (Index HeadDimension, FixedPoint)))  -- ^ K writes (with rotary)
     , Vec NumKeyValueHeads (Signal dom (Maybe (Index HeadDimension, FixedPoint)))  -- ^ V writes
     , Vec NumQueryHeads (Signal dom (Maybe (Index HeadDimension, FixedPoint)))     -- ^ Q BRAM writes
     , Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint)               -- ^ cosVec
     , Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint)               -- ^ sinVec
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom ActivationBramAddr   -- ^ bramRdAddr: drive this to BRAM read port (slot 0)
     )
qkvProjector cycleCounter dramSlaveIn layerIdx inputValid downStreamReady seqPos bramRdData =
  (axiMasterOut, kWrites, vWrites, qBramWrites, cosVec, sinVec, outputValid, readyForInput, bramRdAddr)
 where
  ----------------------------------------------------------------------------
  -- DRAM-backed rmsAttF fetch (once per inputValid; gates QKV start)
  ----------------------------------------------------------------------------
  -- Rising edge: fpVecLoader requires a one-shot trigger (processingControllerFSM
  -- keeps 'enable' high for the entire PROCESSING_RUN state, so we must edge-detect).
  inputValidRise :: Signal dom Bool
  inputValidRise = inputValid .&&. (not <$> register False inputValid)

  (rmsAttAxiMaster, rmsAttVec, rmsAttValid, rmsAttBusy) =
    FPVec.fpVecLoader cycleCounter dramSlaveIn
      inputValidRise
      (Layout.rmsAttAddress <$> layerIdx)

  ----------------------------------------------------------------------------
  -- Serial cos/sin fetch: triggered once rmsAtt completes
  ----------------------------------------------------------------------------
  rmsAttDone :: Signal dom Bool
  rmsAttDone = rmsAttValid .&&. (not <$> register False rmsAttValid)

  (cosAxiMaster, cosVec, cosValid, cosBusy) =
    FPVec.fpVecLoaderDyn cycleCounter dramSlaveIn
      rmsAttDone
      (Layout.rotaryCosAddress <$> seqPos)

  cosDone :: Signal dom Bool
  cosDone = cosValid .&&. (not <$> register False cosValid)

  (sinAxiMaster, sinVec, sinValid, sinBusy) =
    FPVec.fpVecLoaderDyn cycleCounter dramSlaveIn
      cosDone
      (Layout.rotarySinAddress <$> seqPos)

  -- Sequential rmsNorm: starts when rmsAtt weights arrive (rmsAttDone already
  -- exists above as the cos-fetch trigger).  Runs in parallel with cos/sin
  -- fetches; effectiveInputValid waits for whichever finishes last.
  -- bramRdData provides xi element-by-element; rdNext drives the BRAM read address
  -- one cycle ahead so data[counter] arrives in time (1-cycle BRAM latency).
  (rmsNormValid, xNorm, _, rdNext) = rmsNormSeq rmsAttDone bramRdData rmsAttVec

  bramRdAddr :: Signal dom ActivationBramAddr
  bramRdAddr = (slot0BramBase +) . fromIntegral <$> rdNext

  -- Hold latch until all three (rmsAtt + cos + sin) have been fetched AND
  -- rmsNorm has completed for THIS token.
  -- Fire on the rising edge of (sinValid AND rmsNormValid) so it fires exactly
  -- once regardless of which signal arrives last.
  bothReady :: Signal dom Bool
  bothReady = sinValid .&&. rmsNormValid

  pendingInput :: Signal dom Bool
  pendingInput = register False nextPendingInput
   where
    nextPendingInput =
      mux (pendingInput .&&. effectiveInputValid) (pure False) $
      mux inputValidRise (pure True)
      pendingInput

  effectiveInputValid :: Signal dom Bool
  effectiveInputValid = pendingInput .&&. bothReady .&&. (not <$> register False bothReady)

  -- Coordinated consume signal: all heads clear together
  consumeSignal = outputValid .&&. downStreamReady

  ----------------------------------------------------------------------------
  -- AXI arbiter: Q + K + V masters all routed through a single arbiter.
  -- Total: NumQueryHeads + NumKeyValueHeads (K) + NumKeyValueHeads (V)
  -- rmsAtt/cos/sin fetches are serial and muxed in with priority.
  ----------------------------------------------------------------------------
  allAxiMasters :: Vec (NumQueryHeads + NumKeyValueHeads + NumKeyValueHeads) (Master.AxiMasterOut dom)
  allAxiMasters = qAxiMasters ++ (kAxiMasters ++ vAxiMasters)

  allSlaves :: Vec (NumQueryHeads + NumKeyValueHeads + NumKeyValueHeads) (Slave.AxiSlaveIn dom)
  (qkvAxiMasterOut, allSlaves) = ARB.axiArbiterWithRouting dramSlaveIn allAxiMasters

  -- Serial pre-fetch masters have priority; at most one is busy at a time
  axiMasterOut = Master.axiMasterMux rmsAttBusy rmsAttAxiMaster
               $ Master.axiMasterMux cosBusy    cosAxiMaster
               $ Master.axiMasterMux sinBusy    sinAxiMaster
               qkvAxiMasterOut

  -- Split slaves: first NumQueryHeads for Q, next NumKeyValueHeads for K, last for V
  (perQSlaves, kvSlaves)   = splitAt (SNat @NumQueryHeads)    allSlaves
  (perKSlaves, perVSlaves) = splitAt (SNat @NumKeyValueHeads) kvSlaves

  ----------------------------------------------------------------------------
  -- Q DRAM path: FSM sees (pure True), latch clears via consumeSignal
  ----------------------------------------------------------------------------
  qResults :: Vec NumQueryHeads ( Master.AxiMasterOut dom
                                , Signal dom (Maybe (Index HeadDimension, FixedPoint))
                                , Signal dom Bool
                                , Signal dom Bool
                                , QHP.QHeadDebugInfo dom )
  qResults = imap (\headIdx _ ->
      QHP.queryHeadProjector cycleCounter (perQSlaves !! headIdx) layerIdx headIdx
                        effectiveInputValid
                        (pure True)      -- downStreamReady for FSM (always ready for next row)
                        consumeSignal    -- consumeSignal for latch clearing
                        cosVec sinVec xNorm
    ) (repeat () :: Vec NumQueryHeads ())

  qAxiMasters = map (\(axi, _, _, _, _) -> axi) qResults
  qBramWrites = map (\(_, w, _, _, _) -> w) qResults
  qValids     = map (\(_, _, v, _, _) -> v) qResults
  qReadys     = map (\(_, _, _, r, _) -> r) qResults

  ----------------------------------------------------------------------------
  -- KV DRAM path: independent K and V compute paths per head, both via AXI
  ----------------------------------------------------------------------------
  kvResults :: Vec NumKeyValueHeads ( Master.AxiMasterOut dom
                                    , Master.AxiMasterOut dom
                                    , Signal dom (Maybe (Index HeadDimension, FixedPoint))  -- K writes
                                    , Signal dom (Maybe (Index HeadDimension, FixedPoint))  -- V writes
                                    , Signal dom Bool  -- outputValid
                                    , Signal dom Bool  -- readyForInput
                                    )
  kvResults = imap (\kvIdx _ ->
      KVHP.keyValueHeadProjector cycleCounter
        (perKSlaves !! kvIdx)
        (perVSlaves !! kvIdx)
        layerIdx kvIdx
        effectiveInputValid
        (pure True)
        consumeSignal
        cosVec sinVec xNorm
    ) (repeat () :: Vec NumKeyValueHeads ())

  kAxiMasters = map (\(k, _, _, _, _, _) -> k) kvResults
  vAxiMasters = map (\(_, v, _, _, _, _) -> v) kvResults
  kWrites     = map (\(_, _, k, _, _, _) -> k) kvResults
  vWrites     = map (\(_, _, _, v, _, _) -> v) kvResults
  kvValids    = map (\(_, _, _, _, va, _) -> va) kvResults
  kvReadys    = map (\(_, _, _, _, _, r)  -> r)  kvResults

  outputValid   = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
  readyForInput = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)

--------------------------------------------------------------------------------
-- QKV Projection Controller
--------------------------------------------------------------------------------
qkvProjectionController ::
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom FixedPoint             -- ^ bramRdData (BRAM slot 0, 1-cycle latency)
  -> Signal dom (Index SequenceLength)
  -> ( Master.AxiMasterOut dom
     , Vec NumKeyValueHeads (Signal dom (Maybe (Index HeadDimension, FixedPoint)))  -- ^ K writes
     , Vec NumKeyValueHeads (Signal dom (Maybe (Index HeadDimension, FixedPoint)))  -- ^ V writes
     , Vec NumQueryHeads (Signal dom (Maybe (Index HeadDimension, FixedPoint)))     -- ^ Q BRAM writes
     , Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint)               -- ^ cosVec
     , Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint)               -- ^ sinVec
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom ActivationBramAddr   -- ^ bramRdAddr (drive to BRAM read port)
     )
qkvProjectionController cycleCounter dramSlaveIn layerIdx inputValid downStreamReady bramRdData seqPos =
  (axiMasterOut, kWrites, vWrites, qBramWrites, cosVec, sinVec, outputValid, readyForInput, bramRdAddr)
 where
  (enable, outputValid, inReadyRaw) =
    FSM.processingControllerFSM inputValid downStreamReady matVecValid

  (axiMasterOut, kWrites, vWrites, qBramWrites, cosVec, sinVec, matVecValid, projReadyOut, bramRdAddr) =
    qkvProjector cycleCounter dramSlaveIn layerIdx enable downStreamReady
                 seqPos bramRdData

  projReadyOut_d = register True projReadyOut
  readyForInput  = inReadyRaw .&&. projReadyOut_d
