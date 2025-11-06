module LLaMa2.Decoder.LayerStack (
  processActiveLayer, prepareLayerInput, LayerOutput(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), ProcessingState)
import LLaMa2.Types.ModelConfig (NumLayers, ModelDimension)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM (TransformerLayerComponent)
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer (QKVProjectionWeightBuffer)

-- | Output from layer processing (simplified)
data LayerOutput dom = LayerOutput
  { outputData   :: Signal dom LayerData
  , attnDone     :: Signal dom Bool  -- Replaces qkvDone, writeDone, attnDone
  , ffnDone      :: Signal dom Bool
  }

-- | Process active layer with simplified stage enables
processActiveLayer :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom (Index NumLayers)
  -> Signal dom LayerData
  -> Signal dom QKVProjectionWeightBuffer
  -> Signal dom Bool
  -> Vec NumLayers PARAM.TransformerLayerComponent
  -> Signal dom Bool  -- enableAttention (global)
  -> Signal dom Bool  -- enableFFN (global)
  -> Signal dom Bool  -- enableClassifier (global)
  -> LayerOutput dom
processActiveLayer processingState activeLayerIdx inputData weightBuffer useRAM layers
                   enableAttention enableFFN enableClassifier =
  LayerOutput
    { outputData = outputLayerData
    , attnDone   = selectedAttnDone
    , ffnDone    = selectedFfnDone
    }
  where
    -- Create outputs for all layers
    layerOutputs = imap (createLayerOutput inputData) layers
    
    -- Select outputs from the active layer
    (outputLayerData, selectedAttnDone, selectedFfnDone) =
      unbundle $ selectActiveLayer activeLayerIdx layerOutputs

    createLayerOutput :: Signal dom LayerData
                      -> Index NumLayers
                      -> PARAM.TransformerLayerComponent
                      -> ( Signal dom LayerData
                         , Signal dom Bool  -- attnDone
                         , Signal dom Bool  -- ffnDone
                         )
    createLayerOutput inputData' layerIdx layerParams =
      ( outputData'
      , attnDone'
      , ffnDone'
      )
      where
        -- Gate enables for this specific layer
        isThisLayer = activeLayerIdx .==. pure layerIdx
        
        enableAttentionThisLayer = enableAttention .&&. isThisLayer
        enableFFNThisLayer = enableFFN .&&. isThisLayer
        enableClassifierThisLayer = enableClassifier .&&. isThisLayer
        
        ( outputData', attnDone', ffnDone' ) =
          TransformerLayer.transformerLayer
            layerParams
            layerIdx
            processingState
            inputData'
            weightBuffer
            useRAM
            enableAttentionThisLayer
            enableFFNThisLayer
            enableClassifierThisLayer

    selectActiveLayer :: Signal dom (Index NumLayers)
                      -> Vec NumLayers ( Signal dom LayerData
                                       , Signal dom Bool
                                       , Signal dom Bool
                                       )
                      -> Signal dom ( LayerData
                                    , Bool
                                    , Bool
                                    )
    selectActiveLayer idx outputs = (!!) <$> traverse bundle outputs <*> idx

-- | Prepare layer input (unchanged)
prepareLayerInput :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
prepareLayerInput idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
