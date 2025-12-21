module LLaMa2.Layer.Attention.QueryHeadProjector.WeightFetchUnit
  ( WeightFetchIn(..)
  , WeightFetchOut(..)
  , weightFetchUnit
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified Simulation.Parameters as PARAM
import qualified Prelude as P
import Clash.Debug (trace)

--------------------------------------------------------------------------------
-- WeightFetchUnit
-- Coordinates DRAM weight loading via WeightLoader
-- CRITICAL: Gating (rowReqValid .&&. weightReady) happens INSIDE
--           weightReady is NOT exposed as output (avoids circular dependency)
--------------------------------------------------------------------------------
data WeightFetchIn dom = WeightFetchIn
  { wfRowIndex      :: Signal dom (Index HeadDimension)
  , wfRowReqValid   :: Signal dom Bool      -- UNGATED request from compute
  , wfConsumeSignal :: Signal dom Bool
  , wfRowDone       :: Signal dom Bool
  , wfInputValid    :: Signal dom Bool      -- For tracing only
  } deriving (Generic)

data WeightFetchOut dom = WeightFetchOut
  { wfAxiMaster    :: Master.AxiMasterOut dom
  , wfWeightDram   :: Signal dom (RowI8E ModelDimension)
  , wfWeightHC     :: Signal dom (RowI8E ModelDimension)
  , wfWeightValid  :: Signal dom Bool
  , wfIdleReady    :: Signal dom Bool       -- For top-level ready calculation
  -- NOTE: weightReady is NOT exposed - used only internally for gating
  } deriving (Generic)

weightFetchUnit :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> PARAM.DecoderParameters
  -> WeightFetchIn dom
  -> WeightFetchOut dom
weightFetchUnit dramSlaveIn layerIdx headIdx params inputs =
  WeightFetchOut
    { wfAxiMaster    = axiMaster
    , wfWeightDram   = currentRowDramTraced
    , wfWeightHC     = currentRowHC
    , wfWeightValid  = weightValid
    , wfIdleReady    = weightReady  -- Expose as "idleReady" for ready calculation
    }
  where
    (axiMaster, weightLoaderOut, weightValid, weightReady) =
      LOADER.weightLoader dramSlaveIn layerIdx headIdx
                          (wfRowIndex inputs)
                          rowReqPulseTraced             -- Use EDGE-DETECTED request
                          (wfConsumeSignal inputs)
                          (wfRowDone inputs)
                          params

    -- Gate the request INTERNALLY (this is the key!)
    rowReqValidGated = wfRowReqValid inputs .&&. weightReady

    prevRowReqValid = register False rowReqValidGated
    rowReqRise = rowReqValidGated .&&. (not <$> prevRowReqValid)

    -- Also detect row index changes while request is high (for multi-row sequences)
    prevRowIndex = register 0 (wfRowIndex inputs)
    rowIndexChanged = wfRowIndex inputs ./=. prevRowIndex

    -- Pulse on: rising edge of request OR row index change while request high
    rowReqPulse = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)

    -- Trace only on the actual pulse (the real event)
    rowReqPulseTraced = traceRequestPulse layerIdx headIdx
                          rowReqPulse (wfInputValid inputs) weightValid
                          (wfRowIndex inputs)
    ----------------------------------------------------------------------------

    currentRowDramRaw = LOADER.dramRowOut weightLoaderOut
    currentRowHCRaw   = LOADER.hcRowOut weightLoaderOut

    -- Ensure rows don't change while valid
    currentRowDram = LOADER.assertRowStable weightValid currentRowDramRaw
    currentRowHC   = LOADER.assertRowStable weightValid currentRowHCRaw

    currentRowDramTraced = weightMismatchTracer layerIdx weightValid
                             currentRowDram currentRowHC

--------------------------------------------------------------------------------
-- Tracing utilities - EDGE TRIGGERED
--------------------------------------------------------------------------------

-- | Trace only on actual request pulses (not every cycle)
traceRequestPulse :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom Bool -> Signal dom Bool -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
traceRequestPulse layerIdx headIdx pulse inputValid weightValid ri = traced
  where
    traced = go <$> pulse <*> inputValid <*> weightValid <*> ri
    go p iv wv ridx
      | p         = trace (prefix P.++ "REQ_PULSE iv=" P.++ show iv
                          P.++ " wv=" P.++ show wv P.++ " ri=" P.++ show ridx) p
      | otherwise = p
    prefix = "[WeightFetchUnit] L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

-- | Trace weight mismatches
weightMismatchTracer :: Index NumLayers
  -> Signal dom Bool
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
weightMismatchTracer layerIdx valid dram hc = result
  where
    result = check <$> valid <*> dram <*> hc
    check v d h
      | v && (rowExponent d P./= rowExponent h P.|| rowMantissas d P./= rowMantissas h) =
          trace ("[WeightFetchUnit] L" P.++ show layerIdx P.++ " WEIGHT_MISMATCH exp_d=" P.++ show (rowExponent d)
                P.++ " exp_h=" P.++ show (rowExponent h)) d
      | otherwise = d
