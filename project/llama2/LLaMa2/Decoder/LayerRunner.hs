module LLaMa2.Decoder.LayerRunner (
  activeLayerProcessor, layerInputStage, LayerOutputs(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..))
import LLaMa2.Types.ModelConfig (NumLayers, NumKeyValueHeads, SequenceLength, ModelDimension, HeadDimension)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec)

data LayerOutputs dom = LayerOutputs
  { axiMasterOut    :: Master.AxiMasterOut dom              -- single weights AXI master
  , kvAxiMasterOuts :: Vec NumKeyValueHeads (Master.AxiMasterOut dom)
  , qkvOutput       :: Signal dom LayerData
  , attnOutput      :: Signal dom LayerData
  , ffnOutput       :: Signal dom LayerData
  , writeDone       :: Signal dom Bool
  , attnDone        :: Signal dom Bool
  , qkvDone         :: Signal dom Bool
  , ffnDone         :: Signal dom Bool
  , qkvReady        :: Signal dom Bool
  }

{-# NOINLINE activeLayerProcessor #-}
activeLayerProcessor :: forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  )
  =>  Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom                             -- weights DRAM interface
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)      -- KV cache DRAM (one bank per KV head)
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index SequenceLength)
  -> Signal dom LayerData
  -> Signal dom Bool
  -> LayerOutputs dom
activeLayerProcessor cycleCounter dramSlaveIn kvDramSlaves activeLayerIdx seqPos inputData inputValid =
  LayerOutputs
    { axiMasterOut    = singleAxiMaster
    , kvAxiMasterOuts = singleKvMasters
    , qkvOutput       = qkvData
    , attnOutput      = attnData
    , ffnOutput       = ffnData
    , writeDone       = writeDone'
    , attnDone        = attnDone'
    , qkvDone         = qkvDone'
    , ffnDone         = ffnDone'
    , qkvReady        = qkvReady'
    }
  where
    -- Single transformer layer instance reused across all layer passes
    (singleAxiMaster, singleKvMasters, qProj, kProj, vProj, attnOut, ffnOut
      , qkvDone', writeDone', attnDone', ffnDone', qkvReady', _, _, _) =
      TransformerLayer.transformerLayer
        cycleCounter dramSlaveIn kvDramSlaves activeLayerIdx seqPos inputData inputValid

    qkvData  = (\d q k v -> d { queryVectors = q, keyVectors = k, valueVectors = v })
                  <$> inputData <*> qProj <*> kProj <*> vProj
    attnData = (\d attn -> d { attentionOutput = attn }) <$> inputData <*> attnOut
    ffnData  = (\d ffn  -> d { feedForwardOutput = ffn }) <$> inputData <*> ffnOut

layerInputStage :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
layerInputStage idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
