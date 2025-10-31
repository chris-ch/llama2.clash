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

-- | Output from processing the active layer
data LayerOutput dom = LayerOutput
  { outputData   :: Signal dom LayerData
  , writeDone    :: Signal dom Bool
  , attnDone     :: Signal dom Bool
  , qkvDone      :: Signal dom Bool
  , ffnDone      :: Signal dom Bool
  }

-- | Process only the currently active layer
processActiveLayer :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom (Index NumLayers)
  -> Signal dom LayerData
  -> Signal dom QKVProjectionWeightBuffer
  -> Signal dom Bool
  -> Vec NumLayers PARAM.TransformerLayerComponent
  -> Signal dom Bool  -- enableQKV
  -> Signal dom Bool  -- enableWriteKV
  -> Signal dom Bool  -- enableAttend
  -> Signal dom Bool  -- enableFFN
  -> Signal dom Bool  -- enableClassifier
  -> LayerOutput dom
processActiveLayer processingState activeLayerIdx inputData weightBuffer useRAM layers
                   enableQKV enableWriteKV enableAttend enableFFN enableClassifier =
  LayerOutput
    { outputData = outputLayerData
    , writeDone  = selectedWriteDone
    , attnDone   = selectedAttnDone
    , qkvDone    = selectedQkvDone
    , ffnDone    = selectedFfnDone
    }
  where
    -- Create outputs for all layers (only active one will compute)
    layerOutputs = imap createLayerOutput layers
    
    -- Select outputs from the active layer
    (outputLayerData, selectedWriteDone, selectedAttnDone, selectedQkvDone, selectedFfnDone) =
      unbundle $ selectActiveLayer activeLayerIdx layerOutputs
    
    -- Create output for a single layer
    createLayerOutput :: Index NumLayers
                      -> PARAM.TransformerLayerComponent
                      -> ( Signal dom LayerData
                         , Signal dom Bool
                         , Signal dom Bool
                         , Signal dom Bool
                         , Signal dom Bool
                         )
    createLayerOutput layerIdx layerParams =
      ( outputData'
      , writeDone'
      , attnDone'
      , qkvDone'
      , ffnDone'
      )
      where
        ( outputData'
          , writeDone'
          , attnDone'
          , qkvDone'
          , ffnDone'
          ) = TransformerLayer.transformerLayer
                layerParams
                layerIdx
                processingState
                inputData
                weightBuffer
                useRAM
                enableQKV
                enableWriteKV
                enableAttend
                enableFFN
                enableClassifier

    -- Select the output from the active layer using a multiplexer tree
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

-- | Prepare layer input: layer 0 gets embedding, others get previous FFN output
prepareLayerInput :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
prepareLayerInput idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
