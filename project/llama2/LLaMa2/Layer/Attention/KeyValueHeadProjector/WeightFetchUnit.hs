module LLaMa2.Layer.Attention.KeyValueHeadProjector.WeightFetchUnit
  ( WeightFetchIn(..)
  , WeightFetchOut(..)
  , kWeightFetchUnit
  , vWeightFetchUnit
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E)
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master

data WeightFetchIn dom = WeightFetchIn
  { wfRowIndex      :: Signal dom (Index HeadDimension)
  , wfRowReqValid   :: Signal dom Bool
  , wfConsumeSignal :: Signal dom Bool
  , wfRowDone       :: Signal dom Bool
  , wfInputValid    :: Signal dom Bool
  } deriving (Generic)

data WeightFetchOut dom = WeightFetchOut
  { wfAxiMaster    :: Master.AxiMasterOut dom
  , wfWeightDram   :: Signal dom (RowI8E ModelDimension)
  , wfWeightValid  :: Signal dom Bool
  , wfIdleReady    :: Signal dom Bool
  } deriving (Generic)

kWeightFetchUnit :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumKeyValueHeads
  -> WeightFetchIn dom
  -> WeightFetchOut dom
kWeightFetchUnit cycleCounter dramSlaveIn layerIdx kvHeadIdx inputs =
  WeightFetchOut
    { wfAxiMaster    = axiMaster
    , wfWeightDram   = currentRowDram
    , wfWeightValid  = weightValid
    , wfIdleReady    = weightReady
    }
  where
    (axiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
        LOADER.kWeightLoader cycleCounter dramSlaveIn layerIdx kvHeadIdx
                          (wfRowIndex inputs) rowReqPulse
                          (pure True) (wfRowDone inputs)
    weightValid = weightValidRaw
    weightReady = weightReadyRaw
    loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)
    rowReqValidGated = wfRowReqValid inputs .&&. weightReady
    prevRowReqValid  = register False $ mux loaderBecameIdle (pure False) rowReqValidGated
    rowReqRise       = rowReqValidGated .&&. (not <$> prevRowReqValid)
    prevRowIndex     = register 0 (wfRowIndex inputs)
    rowIndexChanged  = wfRowIndex inputs ./=. prevRowIndex
    rowReqPulse      = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)
    currentRowDram = LOADER.assertRowStable weightValid (LOADER.dramRowOut weightLoaderOut)

vWeightFetchUnit :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumKeyValueHeads
  -> WeightFetchIn dom
  -> WeightFetchOut dom
vWeightFetchUnit cycleCounter dramSlaveIn layerIdx kvHeadIdx inputs =
  WeightFetchOut
    { wfAxiMaster    = axiMaster
    , wfWeightDram   = currentRowDram
    , wfWeightValid  = weightValid
    , wfIdleReady    = weightReady
    }
  where
    (axiMaster, weightLoaderOut, weightValidRaw, weightReadyRaw) =
        LOADER.vWeightLoader cycleCounter dramSlaveIn layerIdx kvHeadIdx
                          (wfRowIndex inputs) rowReqPulse
                          (pure True) (wfRowDone inputs)
    weightValid = weightValidRaw
    weightReady = weightReadyRaw
    loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)
    rowReqValidGated = wfRowReqValid inputs .&&. weightReady
    prevRowReqValid  = register False $ mux loaderBecameIdle (pure False) rowReqValidGated
    rowReqRise       = rowReqValidGated .&&. (not <$> prevRowReqValid)
    prevRowIndex     = register 0 (wfRowIndex inputs)
    rowIndexChanged  = wfRowIndex inputs ./=. prevRowIndex
    rowReqPulse      = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)
    currentRowDram = LOADER.assertRowStable weightValid (LOADER.dramRowOut weightLoaderOut)
