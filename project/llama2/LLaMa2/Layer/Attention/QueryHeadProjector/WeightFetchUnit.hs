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

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- WeightFetchUnit
-- Coordinates DRAM weight loading via WeightLoader
--------------------------------------------------------------------------------
data WeightFetchIn dom = WeightFetchIn
  { wfRowIndex      :: Signal dom (Index HeadDimension)
  , wfRowReqValid   :: Signal dom Bool      -- UNGATED request from compute
  , wfConsumeSignal :: Signal dom Bool
  , wfRowDone       :: Signal dom Bool
  , wfInputValid    :: Signal dom Bool      -- (unused now, API compat)
  } deriving (Generic)

data WeightFetchOut dom = WeightFetchOut
  { wfAxiMaster    :: Master.AxiMasterOut dom
  , wfWeightDram   :: Signal dom (RowI8E ModelDimension)
  , wfWeightHC     :: Signal dom (RowI8E ModelDimension)
  , wfWeightValid  :: Signal dom Bool
  , wfIdleReady    :: Signal dom Bool
  } deriving (Generic)

weightFetchUnit :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  ->  Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> PARAM.DecoderParameters
  -> WeightFetchIn dom
  -> WeightFetchOut dom
weightFetchUnit cycleCounter dramSlaveIn layerIdx headIdx params inputs =
  WeightFetchOut
    { wfAxiMaster    = axiMaster
    , wfWeightDram   = currentRowDramChecked
    , wfWeightHC     = currentRowHC
    , wfWeightValid  = weightValid
    , wfIdleReady    = weightReady
    }
  where
    tag = "[WFU L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

    (axiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
        LOADER.weightLoader cycleCounter dramSlaveIn layerIdx headIdx
                          (wfRowIndex inputs)
                          rowReqPulseTraced
                          (wfConsumeSignal inputs)
                          (wfRowDone inputs)
                          params

    -- Trace weight valid and weight ready edges
    weightValid = traceEdgeC cycleCounter (tag P.++ "weightValid") weightValidRaw
    weightReady = traceEdgeC cycleCounter (tag P.++ "weightReady") weightReadyRaw
    
    -- Gate the request INTERNALLY
    rowReqValidGated = wfRowReqValid inputs .&&. weightReady

    prevRowReqValid = register False rowReqValidGated
    rowReqRise = rowReqValidGated .&&. (not <$> prevRowReqValid)

    -- Detect row index changes while request is high
    prevRowIndex = register 0 (wfRowIndex inputs)
    rowIndexChanged = wfRowIndex inputs ./=. prevRowIndex

    -- Pulse on: rising edge of request OR row index change while request high
    rowReqPulse = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)

    -- Simple edge trace
    rowReqPulseTraced = traceEdgeC cycleCounter (tag P.++ "reqPulse") rowReqPulse

    ----------------------------------------------------------------------------
    currentRowDramRaw = LOADER.dramRowOut weightLoaderOut
    currentRowHCRaw   = LOADER.hcRowOut weightLoaderOut

    -- Ensure rows don't change while valid
    currentRowDram = LOADER.assertRowStable weightValid currentRowDramRaw
    currentRowHC   = LOADER.assertRowStable weightValid currentRowHCRaw

    -- Weight mismatch checker (assertion, not a trace)
    currentRowDramChecked = weightMismatchChecker cycleCounter layerIdx weightValid
                              currentRowDram currentRowHC

--------------------------------------------------------------------------------
-- Weight mismatch checker (assertion - kept as explicit check)
--------------------------------------------------------------------------------
weightMismatchChecker :: Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
weightMismatchChecker cycleCounter layerIdx valid dram hc = result
  where
    result = check <$> valid <*> dram <*> hc
    check v d h
      | v && (rowExponent d P./= rowExponent h P.|| rowMantissas d P./= rowMantissas h) =
          trace ("[WFU L" P.++ show layerIdx P.++ "] WEIGHT_MISMATCH exp_d=" P.++ show (rowExponent d)
                P.++ " exp_h=" P.++ show (rowExponent h)) d
      | otherwise = d
