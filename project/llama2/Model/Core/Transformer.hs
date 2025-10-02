module Model.Core.Transformer (
    multiCycleTransformer
) where

import Clash.Prelude
import Model.Core.Transformer.Internal

import Helpers (liftA4)
import Model.Core.Types
  ( LayerData(..)
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
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..), TransformerLayerComponent)
import Data.Maybe (isJust)
import qualified Model.Embedding.PRNG as PRNG
import qualified Model.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs (..)
  )
import qualified Model.Core.Embedding as Embedding
import qualified Model.Layers.Components.Quantized as Quantized (EmbeddingComponentQ(..))
import Model.Numeric.ParamPack (QArray2D)
import Model.Numeric.Types (FixedPoint)

multiCycleTransformer :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom Token
  -> Signal dom Bool            -- ^ inputTokenValid (True while external prompt is used)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token , Signal dom Bool )
multiCycleTransformer decoder cacheOwners inputToken inputTokenValid temperature seed =
  ( selectedToken, readyPulse)
 where
  vocabulary :: QArray2D VocabularySize ModelDimension
  vocabulary = Quantized.vocabularyQ $ modelEmbedding decoder
  
  transformerLayers :: Vec NumLayers TransformerLayerComponent
  transformerLayers  = modelLayers decoder
  
  pipelineController :: PipelineController.PipelineOutputs dom
  pipelineController = PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer inputTokenValid

  layerIndex :: Signal dom (Index NumLayers)
  layerIndex = PipelineController.layerIndex pipelineController

  readyPulse :: Signal dom Bool
  readyPulse = PipelineController.readyPulse pipelineController

  processingState :: Signal dom ProcessingState
  processingState = PipelineController.processingState pipelineController

  tokenSample :: Signal dom Token
  tokenSample = PRNG.tokenSampler readyPulse temperature seed decoder nextLayerData

  feedbackToken :: Signal dom Token
  feedbackToken = regEn 0 readyPulse tokenSample

  selectedToken :: Signal dom Token
  selectedToken = mux inputTokenValid inputToken feedbackToken

  -- Quantized embedding lookup
  tokenEmbedding :: Signal dom (Vec ModelDimension FixedPoint)
  tokenEmbedding = Embedding.embedder vocabulary <$> selectedToken

  layerDataRegister :: Signal dom LayerData
  layerDataRegister = register initialLayerData nextLayerData

  layerInputSelector :: ProcessingState -> LayerData -> Vec ModelDimension FixedPoint -> LayerData
  layerInputSelector ps currentLayerData tokenEmbedding = if processingStage ps == Stage1_ProjectQKV
           then if processingLayer ps == 0
                  then currentLayerData { inputVector = tokenEmbedding }
                  else currentLayerData { inputVector = feedForwardOutput currentLayerData }
           else currentLayerData

  selectedInput :: Signal dom LayerData
  selectedInput = liftA3 layerInputSelector processingState layerDataRegister tokenEmbedding

  transformerLayerProcessor :: ( Signal dom LayerData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
        )
    -> (TransformerLayer.TransformerLayerComponent, Cache.KVRamOwner dom, Index NumLayers)
    -> ( Signal dom LayerData
       , Vec NumLayers (Signal dom Bool)
       , Vec NumLayers (Signal dom Bool)
       )
  transformerLayerProcessor (currentLayerData, wDoneVec, attnDoneVec)
            (layerComp, cacheOwner, layerIndex) =
    let
      (newLayerData, wDone, attnDone, commitC3) =
          TransformerLayer.multiCycleTransformerLayer layerComp cacheOwner layerIndex processingState currentLayerData
      layerDataSelector :: ProcessingState -> LayerData -> LayerData -> LayerData -> LayerData
      layerDataSelector state currentData newD c3D = 
               if processingLayer state == layerIndex
                  then if processingStage state == Stage3_Attend
                         then c3D
                         else newD
                  else currentData
      selectedLayerData :: Signal dom LayerData
      selectedLayerData = liftA4 layerDataSelector processingState currentLayerData newLayerData commitC3
    in  ( selectedLayerData
        , replace layerIndex wDone    wDoneVec
        , replace layerIndex attnDone attnDoneVec
        )

  ( nextLayerData, writeDone , attnDone ) =
      foldl
        transformerLayerProcessor (selectedInput , repeat (pure False) , repeat (pure False))
        (zip3 transformerLayers cacheOwners indicesI)

  writeDoneThisLayer :: Signal dom Bool
  writeDoneThisLayer = (!!) <$> sequenceA writeDone <*> layerIndex

  attnDoneThisLayer :: Signal dom Bool
  attnDoneThisLayer  = (!!) <$> sequenceA attnDone <*> layerIndex
