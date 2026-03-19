module LLaMa2.Decoder.LayerRunner (
  activeLayerProcessor, LayerOutputs(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (ActivationBramAddr)
import LLaMa2.Types.ModelConfig (NumLayers, NumKeyValueHeads, SequenceLength, HeadDimension)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec)

data LayerOutputs dom = LayerOutputs
  { axiMasterOut    :: Master.AxiMasterOut dom
  , kvAxiMasterOuts :: Vec NumKeyValueHeads (Master.AxiMasterOut dom)
  , layerDone       :: Signal dom Bool   -- ^ copy phase complete (slot 3→0 done)
  , readyOut        :: Signal dom Bool   -- ^ layer is idle, ready for new validIn
  , ffnStreamOut    :: Signal dom (Maybe FixedPoint) -- ^ slot 3 elements during copy phase
  }

{-# NOINLINE activeLayerProcessor #-}
activeLayerProcessor :: forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  )
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom                             -- ^ weights DRAM
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)      -- ^ KV cache DRAM (one bank per KV head)
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Maybe (ActivationBramAddr, FixedPoint)) -- ^ slot 0 init write (embedding)
  -> Signal dom Bool                                  -- ^ validIn
  -> LayerOutputs dom
activeLayerProcessor cycleCounter dramSlaveIn kvDramSlaves activeLayerIdx seqPos initWrPort inputValid =
  LayerOutputs
    { axiMasterOut    = singleAxiMaster
    , kvAxiMasterOuts = singleKvMasters
    , layerDone       = done
    , readyOut        = ready
    , ffnStreamOut    = stream
    }
  where
    (singleAxiMaster, singleKvMasters, done, ready, stream) =
      TransformerLayer.transformerLayer
        cycleCounter dramSlaveIn kvDramSlaves activeLayerIdx seqPos initWrPort inputValid
