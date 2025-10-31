module LLaMa2.Decoder.SimplifiedLayerStack (
  processActiveLayer, prepareLayerInput, LayerOutput(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), ProcessingState)
import LLaMa2.Types.ModelConfig (NumLayers, ModelDimension)
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
  -> Signal dom LayerData
  -> Signal dom QKVProjectionWeightBuffer
  -> Signal dom Bool
  -> Vec NumLayers PARAM.TransformerLayerComponent
  -> Signal dom Bool  -- enableQKV (global)
  -> Signal dom Bool  -- enableWriteKV (global)
  -> Signal dom Bool  -- enableAttend (global)
  -> Signal dom Bool  -- enableFFN (global)
  -> LayerOutput dom
processActiveLayer processingState activeLayerIdx inputData weightBuffer useRAM layers
                   enableQKV enableWriteKV enableAttend enableFFN =
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
        
        enableQKVThisLayer = enableQKV .&&. isThisLayer
        enableWriteKVThisLayer = enableWriteKV .&&. isThisLayer
        enableAttendThisLayer = enableAttend .&&. isThisLayer
        enableFFNThisLayer = enableFFN .&&. isThisLayer
        
        ( outputData', writeDone', attnDone', qkvDone', ffnDone' ) =
          TransformerLayer.transformerLayer
            layerParams
            layerIdx
            processingState
            inputData'
            weightBuffer
            useRAM
            enableQKVThisLayer       -- Now layer-specific!
            enableWriteKVThisLayer
            enableAttendThisLayer
            enableFFNThisLayer

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
