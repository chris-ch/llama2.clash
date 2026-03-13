module LLaMa2.Layer.Attention.QueryHeadProjector.WeightFetchUnit
  ( WeightFetchIn(..)
  , WeightFetchOut(..)
  , weightFetchUnit
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E)
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified Prelude as P

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- WeightFetchUnit
-- Coordinates DRAM weight loading via WeightLoader
--------------------------------------------------------------------------------
data WeightFetchIn dom = WeightFetchIn
  { wfRowIndex      :: Signal dom (Index HeadDimension)
  , wfRowReqValid   :: Signal dom Bool
  , wfConsumeSignal :: Signal dom Bool
  , wfRowDone       :: Signal dom Bool
  , wfInputValid    :: Signal dom Bool      -- (unused now, API compat)
  } deriving (Generic)

data WeightFetchOut dom = WeightFetchOut
  { wfAxiMaster    :: Master.AxiMasterOut dom
  , wfWeightDram   :: Signal dom (RowI8E ModelDimension)
  , wfWeightValid  :: Signal dom Bool
  , wfIdleReady    :: Signal dom Bool
  } deriving (Generic)

weightFetchUnit :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  ->  Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumQueryHeads
  -> WeightFetchIn dom
  -> WeightFetchOut dom
weightFetchUnit cycleCounter dramSlaveIn layerIdx headIdx inputs =
  WeightFetchOut
    { wfAxiMaster    = axiMaster
    , wfWeightDram   = currentRowDram
    , wfWeightValid  = weightValid
    , wfIdleReady    = weightReady
    }
  where
    tag = "[WFU H" P.++ show headIdx P.++ "] "

    (axiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
        LOADER.qWeightLoader cycleCounter dramSlaveIn layerIdx headIdx
                          (wfRowIndex inputs)
                          rowReqPulseTraced
                          (pure True)
                          (wfRowDone inputs)

    weightValid = traceEdgeC cycleCounter (tag P.++ "weightValid") weightValidRaw
    weightReady = traceEdgeC cycleCounter (tag P.++ "weightReady") weightReadyRaw

    loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)
    rowReqValidGated = wfRowReqValid inputs .&&. weightReady
    prevRowReqValid  = register False $
        mux loaderBecameIdle (pure False) rowReqValidGated
    rowReqRise       = rowReqValidGated .&&. (not <$> prevRowReqValid)
    prevRowIndex     = register 0 (wfRowIndex inputs)
    rowIndexChanged  = wfRowIndex inputs ./=. prevRowIndex
    rowReqPulse      = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)
    rowReqPulseTraced = traceEdgeC cycleCounter (tag P.++ "reqPulse") rowReqPulse

    currentRowDram = LOADER.assertRowStable weightValid (LOADER.dramRowOut weightLoaderOut)
