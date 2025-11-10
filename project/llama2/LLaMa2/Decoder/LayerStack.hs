module LLaMa2.Decoder.LayerStack (
  processActiveLayer, prepareLayerInput, LayerOutput(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), ProcessingState (..), CycleStage (..))
import LLaMa2.Types.ModelConfig (NumLayers, ModelDimension, SequenceLength)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM (TransformerLayerComponent)
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer (QKVProjectionWeightBuffer)

data LayerOutput dom = LayerOutput
  { outputData   :: Signal dom LayerData
  , writeDone    :: Signal dom Bool
  , attnDone     :: Signal dom Bool
  , qkvDone      :: Signal dom Bool
  , ffnDone      :: Signal dom Bool
  }

processActiveLayer :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index SequenceLength)
  -> Signal dom LayerData
  -> Signal dom QKVProjectionWeightBuffer
  -> Signal dom Bool
  -> Vec NumLayers PARAM.TransformerLayerComponent
  -> Signal dom Bool  -- validIn
  -> LayerOutput dom
processActiveLayer processingState activeLayerIdx seqPos inputData weightBuffer useRAM layers validIn =
  LayerOutput
    { outputData = outputLayerData
    , writeDone  = selectedWriteDone
    , attnDone   = selectedAttnDone
    , qkvDone    = selectedQkvDone
    , ffnDone    = selectedFfnDone
    }
  where
    -- Create outputs for all layers
    layerOutputs = imap (createLayerOutput inputData) layers
    
    -- Select outputs from the active layer
    (outputLayerData, selectedWriteDone, selectedAttnDone, selectedQkvDone, selectedFfnDone) =
      unbundle $ selectActiveLayer activeLayerIdx layerOutputs

    createLayerOutput :: Signal dom LayerData
                      -> Index NumLayers
                      -> PARAM.TransformerLayerComponent
                      -> ( Signal dom LayerData
                         , Signal dom Bool
                         , Signal dom Bool
                         , Signal dom Bool
                         , Signal dom Bool
                         )
    createLayerOutput inputData' layerIdx layerParams =
      ( outputData'
      , writeDone'
      , attnDone'
      , qkvDone'
      , ffnDone'
      )
      where
        -- NEW: Gate enables for this specific layer
        isThisLayer = activeLayerIdx .==. pure layerIdx
        
        validIn' = validIn .&&. isThisLayer
        
        cycleStage = processingStage <$> processingState

        ( qProj, kProj, vProj, attnOut, ffnOut, writeDone', attnDone', qkvDone', ffnDone' ) =
          TransformerLayer.transformerLayer
            layerParams
            seqPos
            cycleStage
            inputData'
            weightBuffer
            useRAM
            validIn'

        -- Now recombine locally, no internal mutation inside transformerLayer
        outputData' =
          mux (cycleStage .==. pure Stage1_ProjectQKV)
            ( (\d q k v -> d { queryVectors = q, keyVectors = k, valueVectors = v })
                <$> inputData' <*> qProj <*> kProj <*> vProj )
            ( mux (cycleStage .==. pure Stage3_Attend)
                ((\d attn -> d { attentionOutput = attn }) <$> inputData' <*> attnOut)
                ( mux (cycleStage .==. pure Stage4_FeedForward)
                    ((\d ffn -> d { feedForwardOutput = ffn }) <$> inputData' <*> ffnOut)
                    inputData'
                )
            )

    selectActiveLayer :: Signal dom (Index NumLayers)
                      -> Vec NumLayers ( Signal dom LayerData
                                       , Signal dom Bool
                                       , Signal dom Bool
                                       , Signal dom Bool
                                       , Signal dom Bool
                                       )
                      -> Signal dom ( LayerData
                                    , Bool
                                    , Bool
                                    , Bool
                                    , Bool
                                    )
    selectActiveLayer idx outputs = (!!) <$> traverse bundle outputs <*> idx

prepareLayerInput :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
prepareLayerInput idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
