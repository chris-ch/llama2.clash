-- | Layer Stack - Processes all transformer layers sequentially
-- Extracted from Decoder to separate layer processing concerns
module LLaMa2.Decoder.LayerStack (
  processLayers, getCurrentLayerFlag, prepareLayerInput
) where

import Clash.Prelude

import LLaMa2.Types.LayerData (LayerData(..), ProcessingState)
import LLaMa2.Types.ModelConfig  (NumLayers, ModelDimension)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types
import LLaMa2.Types.Parameters (TransformerLayerComponent)
import LLaMa2.Layer.Attention.WeightBuffer (QKVWeightBuffer(..))

-- | Type alias for layer completion flags
-- (writeDone, attnDone, qkvDone, qkvReady, ffnDone)
type LayerDoneFlags dom = (Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool)

-- | Process input through all transformer layers sequentially
-- Only the layer indicated by currentLayerIdx performs active computation
processLayers :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom ProcessingState              -- ^ Current processing state
  -> Signal dom (Index NumLayers)            -- ^ Active layer index
  -> Signal dom LayerData                    -- ^ Input layer data
  -> Signal dom QKVWeightBuffer              -- ^ NEW: complete RAM buffer
  -> Signal dom Bool                         -- ^ NEW: useRAM flag
  -> Vec NumLayers TransformerLayerComponent -- ^ All layer parameters
  -> ( Signal dom LayerData                  -- ^ Output layer data
     , Vec NumLayers (LayerDoneFlags dom)    -- ^ Completion flags per layer
     )
processLayers processingState currentLayerIdx inputLayerData weightBuffer useRAM layers =
  (finalLayerData, doneFlagsVec)
  where
    (finalLayerData, layerResults) =
      mapAccumL processOneLayer inputLayerData (imap (,) layers)

    doneFlagsVec = fmap (\(flags, _, _) -> flags) layerResults

    processOneLayer :: Signal dom LayerData
                    -> (Index NumLayers, TransformerLayerComponent)
                    -> (Signal dom LayerData, (LayerDoneFlags dom, Signal dom Bool, Signal dom Bool))
    processOneLayer layerDataIn (layerIdx, layerComponent) =
      let
        isActive = currentLayerIdx .==. pure layerIdx

        ( layerDataOut
          , writeDone
          , attnDone
          , qkvDone
          , _layerDataAfterAttn
          , qkvReady
          , ffnDone
          ) = TransformerLayer.transformerLayer
                layerComponent
                layerIdx
                processingState
                layerDataIn
                weightBuffer        -- NEW
                useRAM              -- NEW

        selectedData = mux isActive layerDataOut layerDataIn

        doneFlags = (writeDone, attnDone, qkvDone, qkvReady, ffnDone)

        validOut = ffnDone
        readyIn = pure True
      in (selectedData, (doneFlags, validOut, readyIn))

-- | Extract completion flag for the currently active layer
getCurrentLayerFlag
  :: Signal dom (Index NumLayers)
  -> Vec NumLayers (Signal dom Bool)
  -> Signal dom Bool
getCurrentLayerFlag currentIdx flags =
  (!!) <$> sequenceA flags <*> currentIdx

-- | Prepare layer input: layer 0 gets embedding, others get previous FFN output
prepareLayerInput 
  :: Index NumLayers 
  -> LayerData 
  -> Vec ModelDimension FixedPoint 
  -> LayerData
prepareLayerInput idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
