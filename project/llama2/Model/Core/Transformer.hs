module Model.Core.Transformer (
    multiCycleTransformer
) where

import Clash.Prelude
import Model.Core.Transformer.Internal

import Helpers (liftA4)
import Model.Core.Types
  ( IntermediateData(..)
  , ProcessingState (..)
  , CycleStage (..)
  , Temperature, Seed
  , Token
  )
import Model.Config
  (  NumLayers, VocabularySize, ModelDimension, SequenceLength
  )
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer
  ( TransformerLayerComponent(..)
  , TransformerDecoderComponent(..)
  , multiCycleTransformerLayer
  )
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..))
import Data.Maybe (isJust)
import qualified Model.Embedding.PRNG as PRNG
import qualified Model.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs (..)
  )

multiCycleTransformer :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom Token
  -> Signal dom Bool            -- ^ inputTokenValid (True while external prompt is used)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token , Signal dom Bool )
multiCycleTransformer decoder cacheOwners inputTokenSignal inputTokenValid temperatureSignal seedSignal =
  ( selectedTokenSignal, PipelineController.readyPulse ctrl)
 where
  embeddingComponent = modelEmbedding decoder
  transformerLayers  = modelLayers decoder

  writeDoneThisLayer = (!!) <$> sequenceA writeDoneVector <*> PipelineController.layerIndex ctrl
  attnDoneThisLayer  = (!!) <$> sequenceA attnDoneVector  <*> PipelineController.layerIndex ctrl

  ctrl = PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer inputTokenValid

  feedbackTokenSignal :: Signal dom Token
  feedbackTokenSignal =
    outputTokenSignal (PipelineController.readyPulse ctrl)
                      temperatureSignal
                      seedSignal
                      decoder
                      nextIntermediateDataSignal

  selectedTokenSignal :: Signal dom Token
  selectedTokenSignal =
    mux inputTokenValid inputTokenSignal feedbackTokenSignal

  -- Quantized embedding lookup
  tokenEmbeddingSignal = embedFromComponent embeddingComponent <$> selectedTokenSignal

  intermediateDataSignal = register initialIntermediateData nextIntermediateDataSignal

  inputLoadedSignal :: Signal dom IntermediateData
  inputLoadedSignal =
    liftA3
      (\ps current tokenEmbedding ->
         if processingStage ps == Stage1_ProjectQKV
           then if processingLayer ps == 0
                  then current { inputVector = tokenEmbedding }
                  else current { inputVector = feedForwardOutput current }
           else current)
      (PipelineController.processingState ctrl) intermediateDataSignal tokenEmbeddingSignal

  layerStep :: ( Signal dom IntermediateData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
        )
    -> (TransformerLayer.TransformerLayerComponent, Cache.KVRamOwner dom, Index NumLayers)
    -> ( Signal dom IntermediateData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       )
  layerStep (currData, wDoneVec, attnDoneVec)
            (layerComp, cacheOwner, lIx) =
    let
      (newData, wDone, attnDone, commitC3) =
          TransformerLayer.multiCycleTransformerLayer layerComp cacheOwner lIx (PipelineController.processingState ctrl) currData
      selectedData =
          liftA4
            (\ps oldD newD c3D ->
               if processingLayer ps == lIx
                  then if processingStage ps == Stage3_Attend
                         then c3D
                         else newD
                  else oldD)
            (PipelineController.processingState ctrl) currData newData commitC3
    in  ( selectedData
        , replace lIx wDone    wDoneVec
        , replace lIx attnDone attnDoneVec
        )

  ( nextIntermediateDataSignal , writeDoneVector , attnDoneVector ) =
      foldl
        layerStep (inputLoadedSignal , repeat (pure False) , repeat (pure False))
        (zip3 transformerLayers cacheOwners indicesI)
