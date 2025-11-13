module LLaMa2.Decoder.LayerStack (
  processActiveLayer, prepareLayerInput, LayerOutputs(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..))
import LLaMa2.Types.ModelConfig (NumLayers, SequenceLength, ModelDimension)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM (TransformerLayerComponent)

-- | All intermediate layer outputs (QKV, Attention, FeedForward)
data LayerOutputs dom = LayerOutputs
  { qkvOutput   :: Signal dom LayerData
  , attnOutput  :: Signal dom LayerData
  , ffnOutput   :: Signal dom LayerData
  , writeDone   :: Signal dom Bool
  , attnDone    :: Signal dom Bool
  , qkvDone     :: Signal dom Bool
  , ffnDone     :: Signal dom Bool
  , qkvReady    :: Signal dom Bool
  }

processActiveLayer :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Index NumLayers)
  -> Signal dom (Index SequenceLength)
  -> Signal dom LayerData
  -> Signal dom Bool  -- inputValid
  -> Vec NumLayers PARAM.TransformerLayerComponent
  -> LayerOutputs dom
processActiveLayer activeLayerIdx seqPos inputData inputValid params =
  LayerOutputs
    { qkvOutput  = selectedQkvOutput
    , attnOutput = selectedAttnOutput
    , ffnOutput  = selectedFfnOutput
    , writeDone  = selectedWriteDone
    , attnDone   = selectedAttnDone
    , qkvDone    = selectedQkvDone
    , ffnDone    = selectedFfnDone
    , qkvReady   = selectedQkvReady
    }
  where
    -- Run all layers in parallel (only one gets inputValid true)
    layerOutputs = imap (layerPipeline inputData) params

    -- Pick outputs for the active layer
    (selectedQkvOutput, selectedAttnOutput, selectedFfnOutput,
     selectedQkvDone, selectedWriteDone, selectedAttnDone,
     selectedFfnDone, selectedQkvReady) =
      unbundle $ selectActiveLayer activeLayerIdx layerOutputs

    layerPipeline :: Signal dom LayerData
                      -> Index NumLayers
                      -> PARAM.TransformerLayerComponent
                      -> ( Signal dom LayerData
                         , Signal dom LayerData
                         , Signal dom LayerData
                         , Signal dom Bool
                         , Signal dom Bool
                         , Signal dom Bool
                         , Signal dom Bool
                         , Signal dom Bool
                         )
    layerPipeline inputData' layerIdx layerParams =
      ( qkvData, attnData, ffnData
      , qkvDone', writeDone', attnDone', ffnDone', qkvReady )
      where
        isThisLayer = activeLayerIdx .==. pure layerIdx
        validIn' = inputValid .&&. isThisLayer

        ( qProj, kProj, vProj, attnOut, ffnOut
          , qkvDone', writeDone', attnDone', ffnDone', qkvReady ) =
            TransformerLayer.transformerLayer
              layerParams
              seqPos
              inputData'
              validIn'

        qkvData  = (\d q k v -> d { queryVectors = q, keyVectors = k, valueVectors = v })
                      <$> inputData' <*> qProj <*> kProj <*> vProj
        attnData = (\d attn -> d { attentionOutput = attn }) <$> inputData' <*> attnOut
        ffnData  = (\d ffn -> d { feedForwardOutput = ffn }) <$> inputData' <*> ffnOut

    selectActiveLayer idx outputs = (!!) <$> traverse bundle outputs <*> idx

prepareLayerInput :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
prepareLayerInput idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
