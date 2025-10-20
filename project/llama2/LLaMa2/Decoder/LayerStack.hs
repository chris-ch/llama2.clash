-- | Layer Stack - Processes all transformer layers sequentially
-- Extracted from Decoder to separate layer processing concerns
module LLaMa2.Decoder.LayerStack
  ( processLayers, getCurrentLayerFlag, layerStack
  , LayerDoneFlags
  ) where

import Clash.Prelude

import LLaMa2.Core.Types (LayerData(..), ProcessingState)
import LLaMa2.Config (NumLayers, ModelDimension)
import LLaMa2.Layer.TransformerLayer (TransformerLayerComponent)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types
import qualified LLaMa2.Decoder.SequenceController as SequenceController

-- | Type alias for layer completion flags
-- (writeDone, attnDone, qkvDone, qkvReady, ffnDone)
type LayerDoneFlags dom = (Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool)

-- | Process input through all transformer layers sequentially
-- Only the layer indicated by currentLayerIdx performs active computation
processLayers
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState              -- ^ Current processing state
  -> Signal dom (Index NumLayers)            -- ^ Active layer index
  -> Signal dom LayerData                    -- ^ Input layer data
  -> Vec NumLayers TransformerLayerComponent -- ^ All layer parameters
  -> ( Signal dom LayerData                  -- ^ Output layer data
     , Vec NumLayers (LayerDoneFlags dom)    -- ^ Completion flags per layer
     )
processLayers processingState currentLayerIdx inputLayerData layers =
  (finalLayerData, doneFlagsVec)
  where
    -- Process each layer, accumulating layer data and collecting flags
    (finalLayerData, layerResults) = 
      mapAccumL processOneLayer inputLayerData (imap (,) layers)
    
    -- Extract done flags from results
    doneFlagsVec = fmap (\(flags, _, _) -> flags) layerResults
    
    -- Process a single layer
    processOneLayer :: Signal dom LayerData 
                    -> (Index NumLayers, TransformerLayerComponent)
                    -> (Signal dom LayerData, (LayerDoneFlags dom, Signal dom Bool, Signal dom Bool))
    processOneLayer layerDataIn (layerIdx, layerComponent) =
      let
        -- Determine if this layer is active
        isActive = currentLayerIdx .==. pure layerIdx
        
        -- Process the layer
        ( layerDataOut
          , writeDone
          , attnDone
          , qkvDone
          , _layerDataAfterAttn  -- Not used in accumulation
          , qkvReady
          , ffnDone
          ) = TransformerLayer.transformerLayer 
                layerComponent 
                layerIdx 
                processingState 
                isActive 
                layerDataIn
        
        -- Only update layer data when this layer is active
        selectedData = mux isActive layerDataOut layerDataIn
        
        -- Package done flags
        doneFlags = (writeDone, attnDone, qkvDone, qkvReady, ffnDone)
        
        -- validOut and readyIn for protocol (currently unused)
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

-- | Simplified layer stack interface - hides LayerData and flag complexity
layerStack
  :: forall dom
   . HiddenClockResetEnable dom
  => Vec NumLayers TransformerLayerComponent
  -> Signal dom (Index NumLayers)
  -> Signal dom Bool  -- ^ inputTokenValid
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec ModelDimension FixedPoint)
     , Signal dom Bool  -- ^ layerComplete (FFN done)
     , Signal dom ProcessingState
     )
layerStack transformerLayers currentLayerIdx inputTokenValid inputVec =
  (layerOutput, layerComplete, processingState)
  where
    -- Internal LayerData state
    layerDataReg = register initialLayerData nextLayerData
    
    layerInput = prepareLayerInput <$> currentLayerIdx <*> layerDataReg <*> inputVec
    
    prepareLayerInput idx currentData embedding
      | idx == 0  = currentData { inputVector = embedding }
      | otherwise = currentData { inputVector = feedForwardOutput currentData }
    
    -- Process layers
    (nextLayerData, doneFlags) = 
      processLayers processingState currentLayerIdx layerInput transformerLayers
    
    -- Extract flags for current layer
    (writeDone, attnDone, qkvDone, _qkvReady, ffnDone) = unzip5 doneFlags
    writeDoneThisLayer = getCurrentLayerFlag currentLayerIdx writeDone
    attnDoneThisLayer  = getCurrentLayerFlag currentLayerIdx attnDone
    qkvDoneThisLayer   = getCurrentLayerFlag currentLayerIdx qkvDone
    ffnDoneThisLayer   = getCurrentLayerFlag currentLayerIdx ffnDone
    
    -- Last layer done signal
    lastLayerDone = (currentLayerIdx .==. pure maxBound) .&&. ffnDoneThisLayer
    
    -- Internal processing state and completion tracking
    pipelineCtrl = SequenceController.pipelineController
        attnDoneThisLayer
        writeDoneThisLayer
        qkvDoneThisLayer
        ffnDoneThisLayer
        lastLayerDone
        inputTokenValid
    
    processingState = SequenceController.processingState pipelineCtrl

    -- Output from layer stack
    layerOutput = feedForwardOutput <$> nextLayerData
    
    -- Layer complete when either FFN done at current layer, or classifier stage finishes
    layerComplete = mux (currentLayerIdx .==. pure maxBound)
                        lastLayerDone  -- At last layer, wait for classifier
                        ffnDoneThisLayer  -- Other layers, just FFN done

-- Need initialLayerData
initialLayerData :: LayerData
initialLayerData = LayerData
  { inputVector       = repeat 0
  , queryVectors      = repeat (repeat 0)
  , keyVectors        = repeat (repeat 0)
  , valueVectors      = repeat (repeat 0)
  , attentionOutput   = repeat 0
  , feedForwardOutput = repeat 0
  }
