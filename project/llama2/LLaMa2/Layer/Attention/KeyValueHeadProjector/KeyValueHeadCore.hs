module LLaMa2.Layer.Attention.KeyValueHeadProjector.KeyValueHeadCore
  ( keyValueHeadCore
  , KVHeadDebugInfo(..)
  ) where

import Clash.Prelude
import qualified Prelude as P

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master

import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector.KVOutputTransactionController as KVOutputTxn
import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector.KVOutputAccumulator as KVOutputAccum
import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector.KVInputTransactionController as KVInputTxn
import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector.KVRowComputeUnit as KVRowCompute
import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector.KVRowScheduler as KVRowSched
import qualified LLaMa2.Layer.Attention.KeyValueHeadProjector.KVWeightFetchUnit as KVWeightFetch

import TraceUtils (traceChangeC, traceEdgeC)

--------------------------------------------------------------------------------
-- Debug Info Record
--------------------------------------------------------------------------------

data KVHeadDebugInfo dom = KVHeadDebugInfo
  { kvhRowIndex        :: Signal dom (Index HeadDimension)
  , kvhState           :: Signal dom OPS.MultiplierState
  , kvhRowDone         :: Signal dom Bool
  , kvhFetchValid      :: Signal dom Bool
  , kvhRowReset        :: Signal dom Bool
  , kvhRowEnable       :: Signal dom Bool
  , kvhKAccumValue     :: Signal dom FixedPoint
  , kvhVAccumValue     :: Signal dom FixedPoint
  , kvhKOut            :: Signal dom (Vec HeadDimension FixedPoint)
  , kvhVOut            :: Signal dom (Vec HeadDimension FixedPoint)
  , kvhRowReqValid     :: Signal dom Bool
  , kvhWeightReady     :: Signal dom Bool
  , kvhWeightValid     :: Signal dom Bool
  } deriving (Generic)

--------------------------------------------------------------------------------
-- Result Checkers (DRAM vs HC verification)
--------------------------------------------------------------------------------

-- | Assert K row results match when rowDone fires
kRowResultChecker :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> Signal dom FixedPoint
  -> Signal dom FixedPoint
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom FixedPoint
kRowResultChecker rowDone rowIdx dramResult hcResult layerIdx' kvHeadIdx' = result
  where
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 nextTokenCnt
    nextTokenCnt = mux (rowDone .&&. (rowIdx .==. pure maxBound)) (tokenCnt + 1) tokenCnt

    result = mux rowDone
                 (check <$> tokenCnt <*> rowIdx <*> dramResult <*> hcResult)
                 dramResult

    check tok ri dr hr
      | dr P.== hr = dr
      | otherwise  = P.error $ "K row result mismatch at token " P.++ show tok
                    P.++ " layer " P.++ show layerIdx'
                    P.++ " kvHead " P.++ show kvHeadIdx'
                    P.++ " row " P.++ show ri 
                    P.++ ": DRAM=" P.++ show dr P.++ " HC=" P.++ show hr

-- | Assert V row results match when rowDone fires
vRowResultChecker :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> Signal dom FixedPoint
  -> Signal dom FixedPoint
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom FixedPoint
vRowResultChecker rowDone rowIdx dramResult hcResult layerIdx' kvHeadIdx' = result
  where
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 nextTokenCnt
    nextTokenCnt = mux (rowDone .&&. (rowIdx .==. pure maxBound)) (tokenCnt + 1) tokenCnt

    result = mux rowDone
                 (check <$> tokenCnt <*> rowIdx <*> dramResult <*> hcResult)
                 dramResult

    check tok ri dr hr
      | dr P.== hr = dr
      | otherwise  = P.error $ "V row result mismatch at token " P.++ show tok
                    P.++ " layer " P.++ show layerIdx'
                    P.++ " kvHead " P.++ show kvHeadIdx'
                    P.++ " row " P.++ show ri 
                    P.++ ": DRAM=" P.++ show dr P.++ " HC=" P.++ show hr

--------------------------------------------------------------------------------
-- Output Vector Checkers
--------------------------------------------------------------------------------

-- | Compare final K/V output vectors
kvOutputChecker :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => String  -- "K" or "V"
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom Bool
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
kvOutputChecker name layerIdx' kvHeadIdx' outputValid dramOut hcOut = result
  where
    everValid = register False (everValid .||. outputValid)
    prevOutputValid = register False outputValid
    checkTrigger = prevOutputValid .&&. everValid

    dramSampled = register (repeat 0) (mux checkTrigger dramOut dramSampled)
    hcSampled   = register (repeat 0) (mux checkTrigger hcOut hcSampled)

    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 (mux checkTrigger (tokenCnt + 1) tokenCnt)

    result = mux checkTrigger (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled) dramOut

    checkPure tok dr hr =
      let pairs = P.zip [0..] (P.zip (toList dr) (toList hr))
          mismatches = P.filter (\(_, (d,h)) -> d P./= h) pairs
      in if P.null mismatches then dr
         else let (i, (d, h)) = P.head mismatches
              in P.error $ name P.++ " output mismatch at token " P.++ show tok
                        P.++ " layer " P.++ show layerIdx'
                        P.++ " kvHead " P.++ show kvHeadIdx'
                        P.++ ": index " P.++ show (i :: Int)
                        P.++ " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")"
                        P.++ " [total mismatches: " P.++ show (P.length mismatches) P.++ "]"

--------------------------------------------------------------------------------
-- KeyValueHeadCore
--------------------------------------------------------------------------------

-- | Core KV head projector with DRAM weight loading
--
-- == Architecture
--
-- @
--                        ┌─────────────────────────────────────────────────────┐
--                        │                  KeyValueHeadCore                   │
--                        │                                                     │
--   inputValid ─────────►│  ┌──────────────┐     ┌──────────────────────┐      │
--   consumeSignal ──────►│  │ InputTxnCtrl │────►│   KVWeightFetchUnit  │      │
--   xHat ───────────────►│  └──────────────┘     │    (K+V weights)     │      │
--                        │         │             └──────────┬───────────┘      │
--                        │         │                        │                  │
--                        │         ▼                        ▼                  │
--                        │  ┌──────────────┐     ┌──────────────────────┐      │
--                        │  │ RowScheduler │────►│   KVRowComputeUnit   │      │
--                        │  └──────────────┘     │   (K+V in parallel)  │      │
--                        │                       └──────────┬───────────┘      │
--                        │                                  │                  │
--                        │                                  ▼                  │
--                        │                       ┌──────────────────────┐      │
--                        │                       │  KVOutputAccumulator │      │
--                        │                       │   (K vec + V vec)    │      │
--                        │                       └──────────┬───────────┘      │
--                        │                                  │                  │
--                        │  ┌──────────────┐                │                  │
--   outputValid ◄────────│──│OutputTxnCtrl │◄───────────────┘                  │
--   kOut ◄───────────────│──┤              │                                   │
--   vOut ◄───────────────│──┘              │                                   │
--                        │                                                     │
--   axiMaster ◄──────────│  (to AXI arbiter)                                   │
--                        └─────────────────────────────────────────────────────┘
-- @
--
keyValueHeadCore :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)               -- ^ cycleCounter
  -> Slave.AxiSlaveIn dom                   -- ^ AXI slave (from arbiter)
  -> Index NumLayers                        -- ^ Layer index (static)
  -> Index NumKeyValueHeads                 -- ^ KV head index (static)
  -> Signal dom Bool                        -- ^ inputValid
  -> Signal dom Bool                        -- ^ downStreamReady
  -> Signal dom Bool                        -- ^ consumeSignal
  -> Signal dom (Vec ModelDimension FixedPoint)  -- ^ xHat (normalized input)
  -> PARAM.DecoderParameters                -- ^ Model parameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)  -- K output
     , Signal dom (Vec HeadDimension FixedPoint)  -- V output
     , Signal dom Bool                             -- outputValid
     , Signal dom Bool                             -- readyForInput
     , KVHeadDebugInfo dom
     )
keyValueHeadCore cycleCounter dramSlaveIn layerIdx kvHeadIdx inputValid downStreamReady consumeSignal xHat params =
  ( KVWeightFetch.kvwfAxiMaster weightFetch
  , kOutFinal
  , vOutFinal
  , KVOutputTxn.kvotcOutputValid outputTxn
  , readyForInput
  , debugInfo
  )
  where
    tag = "[KVHC L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

    ----------------------------------------------------------------------------
    -- Row Index Register
    ----------------------------------------------------------------------------
    
    rowIndex :: Signal dom (Index HeadDimension)
    rowIndex = traceChangeC cycleCounter (tag P.++ "rowIndex") $ register 0 nextRowIndex

    -- RowScheduler computes next index (combinatorial)
    rsIn = KVRowSched.KVRowSchedulerIn
      { KVRowSched.kvrsRowDone       = rowDone
      , KVRowSched.kvrsOutputValid   = KVOutputTxn.kvotcOutputValid outputTxn
      , KVRowSched.kvrsConsumeSignal = consumeSignal
      , KVRowSched.kvrsCurrentIndex  = rowIndex
      }

    rowSched = KVRowSched.kvRowScheduler rsIn
    nextRowIndex = KVRowSched.kvrsNextRowIndex rowSched

    ----------------------------------------------------------------------------
    -- Input Transaction Controller
    ----------------------------------------------------------------------------
    
    inputTxn = KVInputTxn.kvInputTransactionController cycleCounter layerIdx kvHeadIdx rowIndex
                 KVInputTxn.KVInputTransactionIn
                   { KVInputTxn.kvitcInputValid      = inputValid
                   , KVInputTxn.kvitcOutputValid     = KVOutputTxn.kvotcOutputValid outputTxn
                   , KVInputTxn.kvitcDownStreamReady = downStreamReady
                   , KVInputTxn.kvitcConsumeSignal   = consumeSignal
                   }

    inputValidLatched = KVInputTxn.kvitcLatchedValid inputTxn

    ----------------------------------------------------------------------------
    -- Output Transaction Controller
    ----------------------------------------------------------------------------
    
    outputTxn = KVOutputTxn.kvOutputTransactionController cycleCounter layerIdx kvHeadIdx rowIndex downStreamReady
                  KVOutputTxn.KVOutputTransactionIn
                    { KVOutputTxn.kvotcAllDone       = KVRowCompute.kvrcAllDone compute
                    , KVOutputTxn.kvotcConsumeSignal = consumeSignal
                    }

    -- Effective row index that resets combinationally
    effectiveRowIndex :: Signal dom (Index HeadDimension)
    effectiveRowIndex = mux (KVRowSched.kvrsOutputValid rsIn .&&. KVRowSched.kvrsConsumeSignal rsIn)
                            (pure 0)
                            rowIndex

    ----------------------------------------------------------------------------
    -- Weight Fetch Unit
    ----------------------------------------------------------------------------
    
    weightFetch = KVWeightFetch.kvWeightFetchUnit cycleCounter dramSlaveIn layerIdx kvHeadIdx params
                    KVWeightFetch.KVWeightFetchIn
                      { KVWeightFetch.kvwfRowIndex      = effectiveRowIndex
                      , KVWeightFetch.kvwfRowReqValid   = KVRowCompute.kvrcFetchReq compute
                      , KVWeightFetch.kvwfConsumeSignal = consumeSignal
                      , KVWeightFetch.kvwfRowDone       = KVRowCompute.kvrcRowDone compute
                      , KVWeightFetch.kvwfInputValid    = inputValid
                      }

    currentKRowDram = KVWeightFetch.kvwfKWeightDram weightFetch
    currentVRowDram = KVWeightFetch.kvwfVWeightDram weightFetch
    currentKRowHC   = KVWeightFetch.kvwfKWeightHC weightFetch
    currentVRowHC   = KVWeightFetch.kvwfVWeightHC weightFetch
    weightValid     = KVWeightFetch.kvwfWeightValid weightFetch
    weightReady     = KVWeightFetch.kvwfIdleReady weightFetch

    ----------------------------------------------------------------------------
    -- Row Compute Unit (K and V in parallel)
    ----------------------------------------------------------------------------
    
    -- Track if we just consumed (to prevent immediate restart)
    justConsumed :: Signal dom Bool
    justConsumed = register False consumeSignal

    -- Don't allow compute to restart while outputValid is high OR just after consume
    effectiveInputValid = inputValidLatched .&&.
                          (not <$> KVOutputTxn.kvotcOutputValid outputTxn) .&&.
                          (not <$> justConsumed)

    compute = KVRowCompute.kvRowComputeUnit cycleCounter
                KVRowCompute.KVRowComputeIn
                  { KVRowCompute.kvrcInputValid      = effectiveInputValid
                  , KVRowCompute.kvrcWeightValid     = weightValid
                  , KVRowCompute.kvrcDownStreamReady = downStreamReady
                  , KVRowCompute.kvrcRowIndex        = rowIndex
                  , KVRowCompute.kvrcKWeightDram     = currentKRowDram
                  , KVRowCompute.kvrcVWeightDram     = currentVRowDram
                  , KVRowCompute.kvrcKWeightHC       = currentKRowHC
                  , KVRowCompute.kvrcVWeightHC       = currentVRowHC
                  , KVRowCompute.kvrcColumn          = xHat
                  }

    readyForInput = KVRowCompute.kvrcIdleReady compute .&&. weightReady

    ----------------------------------------------------------------------------
    -- Row Done
    ----------------------------------------------------------------------------
    
    rowDone = traceEdgeC cycleCounter (tag P.++ "rowDone") $ KVRowCompute.kvrcRowDone compute

    ----------------------------------------------------------------------------
    -- Row Result Checkers (DRAM vs HC)
    ----------------------------------------------------------------------------
    
    kDramRowResultChecked = kRowResultChecker
      (KVRowCompute.kvrcRowDone compute) rowIndex
      (KVRowCompute.kvrcKResult compute)
      (KVRowCompute.kvrcKResultHC compute)
      layerIdx kvHeadIdx

    vDramRowResultChecked = vRowResultChecker
      (KVRowCompute.kvrcRowDone compute) rowIndex
      (KVRowCompute.kvrcVResult compute)
      (KVRowCompute.kvrcVResultHC compute)
      layerIdx kvHeadIdx

    ----------------------------------------------------------------------------
    -- Output Accumulator
    ----------------------------------------------------------------------------
    
    outputAccum = KVOutputAccum.kvOutputAccumulator cycleCounter layerIdx kvHeadIdx
                    KVOutputAccum.KVOutputAccumIn
                      { KVOutputAccum.kvoaRowDone     = KVRowCompute.kvrcRowDone compute
                      , KVOutputAccum.kvoaRowIndex    = rowIndex
                      , KVOutputAccum.kvoaKResult     = kDramRowResultChecked
                      , KVOutputAccum.kvoaVResult     = vDramRowResultChecked
                      , KVOutputAccum.kvoaKResultHC   = KVRowCompute.kvrcKResultHC compute
                      , KVOutputAccum.kvoaVResultHC   = KVRowCompute.kvrcVResultHC compute
                      }

    kOut   = KVOutputAccum.kvoaKOutput outputAccum
    vOut   = KVOutputAccum.kvoaVOutput outputAccum
    kOutHC = KVOutputAccum.kvoaKOutputHC outputAccum
    vOutHC = KVOutputAccum.kvoaVOutputHC outputAccum

    -- Final output verification
    kOutFinal = kvOutputChecker "K" layerIdx kvHeadIdx 
                                (KVOutputTxn.kvotcOutputValid outputTxn) kOut kOutHC
    vOutFinal = kvOutputChecker "V" layerIdx kvHeadIdx 
                                (KVOutputTxn.kvotcOutputValid outputTxn) vOut vOutHC

    ----------------------------------------------------------------------------
    -- Debug Info
    ----------------------------------------------------------------------------
    
    debugInfo = KVHeadDebugInfo
      { kvhRowIndex     = rowIndex
      , kvhState        = KVRowCompute.kvrcMultState compute
      , kvhRowDone      = KVRowCompute.kvrcRowDone compute
      , kvhFetchValid   = weightValid
      , kvhRowReset     = KVRowCompute.kvmdRowReset (KVRowCompute.kvrcDebug compute)
      , kvhRowEnable    = KVRowCompute.kvmdRowEnable (KVRowCompute.kvrcDebug compute)
      , kvhKAccumValue  = KVRowCompute.kvmdAccValueK (KVRowCompute.kvrcDebug compute)
      , kvhVAccumValue  = KVRowCompute.kvmdAccValueV (KVRowCompute.kvrcDebug compute)
      , kvhKOut         = kOut
      , kvhVOut         = vOut
      , kvhRowReqValid  = KVRowCompute.kvrcFetchReq compute
      , kvhWeightReady  = weightReady
      , kvhWeightValid  = weightValid
      }
