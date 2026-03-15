module LLaMa2.Decoder.LayerRunner (
  activeLayerProcessor, layerInputStage, LayerOutputs(..), layerRunnerTop
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

activeLayerProcessorFlat
  :: ( HiddenClockResetEnable dom
     , KnownNat (WordsPerFPVec HeadDimension)
     )
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index SequenceLength)
  -> Signal dom LayerData
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , Vec NumKeyValueHeads (Master.AxiMasterOut dom)
     , Signal dom LayerData   -- qkvOutput
     , Signal dom LayerData   -- attnOutput
     , Signal dom LayerData   -- ffnOutput
     , Signal dom Bool        -- writeDone
     , Signal dom Bool        -- attnDone
     , Signal dom Bool        -- qkvDone
     , Signal dom Bool        -- ffnDone
     , Signal dom Bool        -- qkvReady
     )
activeLayerProcessorFlat cc dram kv li sp inp iv =
  ( axiMasterOut lo, kvAxiMasterOuts lo
  , qkvOutput lo, attnOutput lo, ffnOutput lo
  , writeDone lo, attnDone lo, qkvDone lo, ffnDone lo, qkvReady lo )
  where lo = activeLayerProcessor cc dram kv li sp inp iv

{-# ANN layerRunnerTop
  (Synthesize
    { t_name   = "layer_runner"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "cycle_counter"
        , PortProduct "weights_dram" []
        , PortProduct "kv_dram"      []
        , PortName "layer_idx"
        , PortName "seq_pos"
        , PortName "input_data"
        , PortName "input_valid"
        ]
    , t_output = PortProduct ""
        [ PortProduct "weights_axi" []
        , PortProduct "kv_axi"      []
        , PortName "qkv_output"
        , PortName "attn_output"
        , PortName "ffn_output"
        , PortName "write_done"
        , PortName "attn_done"
        , PortName "qkv_done"
        , PortName "ffn_done"
        , PortName "qkv_ready"
        ]
    }) #-}
layerRunnerTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Unsigned 32)
  -> Slave.AxiSlaveIn System
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn System)
  -> Signal System (Index NumLayers)
  -> Signal System (Index SequenceLength)
  -> Signal System LayerData
  -> Signal System Bool
  -> ( Master.AxiMasterOut System
     , Vec NumKeyValueHeads (Master.AxiMasterOut System)
     , Signal System LayerData
     , Signal System LayerData
     , Signal System LayerData
     , Signal System Bool
     , Signal System Bool
     , Signal System Bool
     , Signal System Bool
     , Signal System Bool
     )
layerRunnerTop = exposeClockResetEnable activeLayerProcessorFlat

layerInputStage :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
layerInputStage idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
