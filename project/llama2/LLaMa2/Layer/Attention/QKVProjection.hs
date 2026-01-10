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
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import Simulation.Parameters (DecoderParameters(..))
import qualified LLaMa2.Memory.AXI.Arbiter as ARB
import qualified LLaMa2.Layer.Attention.QueryHeadProjector as QHP (queryHeadProjector)
import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector as KVHP
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.QueryHeadCore as QHC

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
-- == Input Signals
--
-- [@inputValid@] Start signal for all heads.
--                Directly passed to each head's inputValid.
--
-- [@downStreamReady@] Acknowledgment from downstream.
--                     __CRITICAL__: In the working HC-path version, this is
--                     passed DIRECTLY to each head. Each head clears its
--                     output latch independently when downStreamReady arrives.
--
-- [@seqPos@] Sequence position for rotary encoding.
--
-- [@xVec@] Input activations (Vec ModelDimension FixedPoint).
--          Normalized internally using RMS norm before projection.
--
-- [@params@] Full model parameters.
--
-- == Output Signals
--
-- [@qkvOut@] Bundled output: (Vec NumQueryHeads qVec, Vec NumKVHeads kVec, Vec NumKVHeads vVec)
--
-- [@outputValid@] True when ALL heads have completed.
--                 Computed as: AND of all individual head valids.
--
-- [@readyForInput@] True when ALL heads are ready for new input.
--                   Computed as: AND of all individual head readys.
--
-- == Coordination Strategy (Working HC-path Version)
--
-- In the current working version, all heads receive `downStreamReady` directly:
--
-- @
-- qResults = map (qHead params) indicesI
--   where
--     qHead params' headIdx = 
--       queryHeadProjector dramSlaveIn layerIdx headIdx
--                         inputValid downStreamReady seqPos xNorm params'
--                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
--                         Direct downStreamReady to each head
-- @
--
-- This means:
-- 1. All heads start simultaneously when inputValid arrives
-- 2. Heads may finish at different times (due to AXI arbitration in DRAM path)
-- 3. When downStreamReady arrives, ALL heads clear their latches simultaneously
-- 4. Combined outputValid = AND of all head valids
--
-- The key insight: even though heads might finish at slightly different times,
-- they all CLEAR together when downStreamReady arrives, ensuring clean handoff.
--
-- == Alternative: consumeSignal Coordination (for DRAM path)
--
-- For proper DRAM arbitration, heads need coordinated clearing:
--
-- @
-- consumeSignal = outputValid .&&. downStreamReady
--
-- qResults' = imap (\headIdx _ ->
--     queryHeadProjector (perHeadSlaves !! headIdx) layerIdx headIdx
--                       inputValid consumeSignal seqPos xNorm params
--                       ^^^^^^^^^^^^
--                       consumeSignal instead of downStreamReady
--   ) (repeat () :: Vec NumQueryHeads ())
-- @
--
-- This ensures:
-- 1. Heads only clear when ALL heads are done AND downstream ready
-- 2. Prevents early-finishing heads from restarting before others complete
-- 3. Required for proper AXI arbiter operation
--
-- __REQUIREMENT__: When using consumeSignal, outputValidLatch MUST have
-- CLR priority over SET (as documented in queryHeadMatrixMultiplier).
--
-- == Usage Notes
--
-- 1. Start processing by asserting inputValid for at least 1 cycle.
--
-- 2. Wait for outputValid before reading qkvOut.
--
-- 3. Assert downStreamReady to acknowledge and prepare for next token.
--
-- 4. Ensure proper handshake: don't assert new inputValid until previous
--    operation is acknowledged.
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
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHC.QHeadDebugInfo dom
     )
qkvProjector cycleCounter dramSlaveIn layerIdx inputValid downStreamReady seqPos xVec params =
  (axiMasterOut, qkvOut, outputValid, readyForInput, head0Debug)
 where
  layerParams = modelLayers params !! layerIdx
  mhaParams = PARAM.multiHeadAttention layerParams
  xNorm = rmsNormFwFix <$> xVec <*> pure (PARAM.rmsAttF mhaParams)

  -- Get global rotary once
  rotary = PARAM.rotaryEncoding params

  -- AXI arbiter setup (mutual recursion handled by lazy evaluation)
  qAxiMasters :: Vec NumQueryHeads (Master.AxiMasterOut dom)
  perHeadSlaves :: Vec NumQueryHeads (Slave.AxiSlaveIn dom)
  
  (axiMasterOut, perHeadSlaves) = ARB.axiArbiterWithRouting cycleCounter dramSlaveIn qAxiMasters

  -- Define coordinated consume signal AFTER outputValid is computed
  consumeSignal = outputValid .&&. downStreamReady

  -- Working version: direct connection, each head gets downStreamReady for FSM
  -- and consumeSignal for latch clearing
  qResults :: Vec NumQueryHeads (Master.AxiMasterOut dom, Signal dom (Vec HeadDimension FixedPoint), Signal dom Bool, Signal dom Bool, QHC.QHeadDebugInfo dom)
  qResults = imap (\headIdx _ ->
      QHP.queryHeadProjector cycleCounter (perHeadSlaves !! headIdx) layerIdx headIdx
                        inputValid 
                        (pure True)      -- downStreamReady for FSM (always ready for next row)
                        consumeSignal    -- consumeSignal for latch clearing
                        seqPos xNorm params
    ) (repeat () :: Vec NumQueryHeads ())

  head0Debug = head qDebugInfos
  qAxiMasters = map (\(axi, _, _, _, _) -> axi) qResults
  qVecs       = map (\(_, q, _, _, _) -> q) qResults
  qValids     = map (\(_, _, v, _, _) -> v) qResults
  qReadys     = map (\(_, _, _, r, _) -> r) qResults
  qDebugInfos = map (\(_, _, _, _, d) -> d) qResults

  kvResults = map kvHead indicesI
   where
    kvHead kvIdx =
      let kvHeadParams = PARAM.kvHeads mhaParams !! kvIdx  -- Get actual KV head
      in KVHP.keyValueHeadProjector inputValid downStreamReady seqPos xNorm kvHeadParams rotary
  
  kVecs    = map (\(k, _, _, _) -> k) kvResults
  vVecs    = map (\(_, v, _, _) -> v) kvResults
  kvValids = map (\(_, _, v, _) -> v) kvResults
  kvReadys = map (\(_, _, _, r) -> r) kvResults
  outputValid = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
  readyForInput = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)
  qkvOut = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)

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
  -> PARAM.DecoderParameters
  -> Signal dom (Index SequenceLength)
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHC.QHeadDebugInfo dom
     )
qkvProjectionController cycleCounter dramSlaveIn layerIdx inputValid downStreamReady input params seqPos =
  (axiMasterOut, result, outputValid, readyForInput, debugInfo)
 where
  (enable, outputValid, inReadyRaw) =
    FSM.processingControllerFSM inputValid downStreamReady matVecValid

  (axiMasterOut, result, matVecValid, projReadyOut, debugInfo) =
    qkvProjector cycleCounter dramSlaveIn layerIdx enable downStreamReady
                 seqPos input params

  projReadyOut_d = register True projReadyOut
  readyForInput  = inReadyRaw .&&. projReadyOut_d
