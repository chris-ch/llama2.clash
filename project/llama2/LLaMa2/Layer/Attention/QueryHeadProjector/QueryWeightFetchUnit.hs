module LLaMa2.Layer.Attention.QueryHeadProjector.QueryWeightFetchUnit
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

import TraceUtils (traceEdgeC, traceWhenC)

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
                          (pure True)           -- ← Always ready for next row
                          (wfRowDone inputs)
                          params

    -- Trace weight valid and weight ready edges
    weightValid = traceEdgeC cycleCounter (tag P.++ "weightValid") weightValidRaw
    weightReady = traceEdgeC cycleCounter (tag P.++ "weightReady") weightReadyRaw
    
    -- Reset edge detection when loader transitions back to idle
    loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)

    -- Gate the request INTERNALLY  
    rowReqValidGated = wfRowReqValid inputs .&&. weightReady

    -- Track previous state, but RESET when loader becomes idle
    prevRowReqValid = register False $ 
        mux loaderBecameIdle (pure False) rowReqValidGated

    rowReqRise = rowReqValidGated .&&. (not <$> prevRowReqValid)

    -- Also reset prevRowIndex tracking
    prevRowIndex = register 0 $ 
        mux loaderBecameIdle (wfRowIndex inputs) (wfRowIndex inputs)
        
    rowIndexChanged = wfRowIndex inputs ./=. prevRowIndex

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
weightMismatchChecker
  :: forall dom . ( KnownNat ModelDimension)
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
weightMismatchChecker cycleCounter layerIdx valid dram hc =
  traceWhenC
    cycleCounter
    ("WFU L" P.++ show layerIdx P.++ " WEIGHT_MISMATCH")
    mismatchCondition
    dram
  where
    -- Version A – clearest and most common style
    mismatchCondition :: Signal dom Bool
    mismatchCondition =
         valid
      .&&. (   (rowExponent <$> dram) ./=. (rowExponent <$> hc)
           .||. (rowMantissas <$> dram) ./=. (rowMantissas <$> hc) )
