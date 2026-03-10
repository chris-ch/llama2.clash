module LLaMa2.Layer.Attention.QKVProjection
  (
    qkvProjectionController
  ) where

import Clash.Prelude


import LLaMa2.Types.ModelConfig
    ( HeadDimension,
      ModelDimension,
      NumKeyValueHeads,
      NumLayers,
      NumQueryHeads,
      SequenceLength )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
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
qkvProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHP.QHeadDebugInfo dom
     )
qkvProjector cycleCounter dramSlaveIn layerIdx inputValid downStreamReady seqPos xVec =
  (axiMasterOut, qkvOut, outputValid, readyForInput, head0Debug)
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
      (pure (Layout.rmsAttAddress layerIdx))

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

  -- Rising edge of sinValid: fires exactly once when sin fetch completes.
  -- Using the rising edge (not the level) prevents immediate firing on token N+1
  -- when sinValid is still True from token N's completed fetch.
  sinDone :: Signal dom Bool
  sinDone = sinValid .&&. (not <$> register False sinValid)

  -- Hold latch until all three (rmsAtt + cos + sin) have been fetched for THIS token.
  -- Cleared by sinDone (rising edge) so it stays False if sinValid is already True.
  pendingInput :: Signal dom Bool
  pendingInput = register False nextPendingInput
   where
    nextPendingInput =
      mux (pendingInput .&&. sinDone) (pure False) $
      mux inputValidRise (pure True)
      pendingInput

  effectiveInputValid :: Signal dom Bool
  effectiveInputValid = pendingInput .&&. sinDone

  -- Pre-normalise with DRAM-fetched RMS weights
  xNorm = rmsNormFwFix <$> xVec <*> rmsAttVec

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
                                , Signal dom (Vec HeadDimension FixedPoint)
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

  head0Debug  = head qDebugInfos
  qAxiMasters = map (\(axi, _, _, _, _) -> axi) qResults
  qVecs       = map (\(_, q, _, _, _) -> q) qResults
  qValids     = map (\(_, _, v, _, _) -> v) qResults
  qReadys     = map (\(_, _, _, r, _) -> r) qResults
  qDebugInfos = map (\(_, _, _, _, d) -> d) qResults

  ----------------------------------------------------------------------------
  -- KV DRAM path: independent K and V compute paths per head, both via AXI
  ----------------------------------------------------------------------------
  kvResults :: Vec NumKeyValueHeads ( Master.AxiMasterOut dom  -- K AXI master
                                    , Master.AxiMasterOut dom  -- V AXI master
                                    , Signal dom (Vec HeadDimension FixedPoint)  -- K (with rotary)
                                    , Signal dom (Vec HeadDimension FixedPoint)  -- V
                                    , Signal dom Bool  -- outputValid
                                    , Signal dom Bool  -- readyForInput
                                    )
  kvResults = imap (\kvIdx _ ->
      KVHP.keyValueHeadProjector cycleCounter
        (perKSlaves !! kvIdx)   -- K DRAM slave
        (perVSlaves !! kvIdx)   -- V DRAM slave
        layerIdx kvIdx
        effectiveInputValid
        (pure True)             -- downStreamReady for FSM (always ready for next row)
        consumeSignal           -- consumeSignal for coordinated latch clearing
        cosVec sinVec xNorm
    ) (repeat () :: Vec NumKeyValueHeads ())

  kAxiMasters = map (\(k, _, _, _, _, _) -> k) kvResults
  vAxiMasters = map (\(_, v, _, _, _, _) -> v) kvResults
  kVecs       = map (\(_, _, k, _, _, _) -> k) kvResults
  vVecs       = map (\(_, _, _, v, _, _) -> v) kvResults
  kvValids    = map (\(_, _, _, _, va, _) -> va) kvResults
  kvReadys    = map (\(_, _, _, _, _, r)  -> r)  kvResults

  outputValid   = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
  readyForInput = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)
  qkvOut        = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)

--------------------------------------------------------------------------------
-- QKV Projection Controller
--------------------------------------------------------------------------------
qkvProjectionController ::
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (Index SequenceLength)
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHP.QHeadDebugInfo dom
     )
qkvProjectionController cycleCounter dramSlaveIn layerIdx inputValid downStreamReady input seqPos =
  (axiMasterOut, result, outputValid, readyForInput, debugInfo)
 where
  (enable, outputValid, inReadyRaw) =
    FSM.processingControllerFSM inputValid downStreamReady matVecValid

  (axiMasterOut, result, matVecValid, projReadyOut, debugInfo) =
    qkvProjector cycleCounter dramSlaveIn layerIdx enable downStreamReady
                 seqPos input

  projReadyOut_d = register True projReadyOut
  readyForInput  = inReadyRaw .&&. projReadyOut_d
