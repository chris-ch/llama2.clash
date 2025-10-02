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
  (  NumLayers, VocabularySize, ModelDimension
  )
import qualified Model.Memory.KVCacheBank as Cache (KVRamOwner)
import qualified Model.Layers.TransformerLayer as TransformerLayer
  ( TransformerDecoderComponent(..)
  , multiCycleTransformerLayer
  )
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..), TransformerLayerComponent)
import qualified Model.Embedding.PRNG as PRNG (tokenSampler)
import qualified Model.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs (..)
  )
import qualified Model.Core.Embedding as Embedding (embedder)
import qualified Model.Layers.Components.Quantized as Quantized (EmbeddingComponentQ(..))
import Model.Numeric.ParamPack (QArray2D)
import Model.Numeric.Types (FixedPoint)

type LayerProcessorData dom = (Signal dom LayerData, Vec NumLayers (Signal dom Bool, Signal dom Bool))

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
  layerInputSelector ps currentLayerData tokenEmbed
    | processingStage ps /= Stage1_ProjectQKV = currentLayerData
    | processingLayer ps == 0                 = currentLayerData { inputVector = tokenEmbed }
    | otherwise                               = currentLayerData { inputVector = feedForwardOutput currentLayerData }

  selectedInput :: Signal dom LayerData
  selectedInput = liftA3 layerInputSelector processingState layerDataRegister tokenEmbedding

  (nextLayerData, doneFlags) = pipelineProcessor processingState selectedInput cacheOwners transformerLayers

  (writeDone, attnDone) = unzip doneFlags

  writeDoneThisLayer :: Signal dom Bool
  writeDoneThisLayer = (!!) <$> sequenceA writeDone <*> layerIndex

  attnDoneThisLayer :: Signal dom Bool
  attnDoneThisLayer  = (!!) <$> sequenceA attnDone <*> layerIndex

pipelineProcessor :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom LayerData
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Vec NumLayers TransformerLayerComponent
  -> LayerProcessorData dom
pipelineProcessor processingState initLayerData cacheOwners transformerLayers =
  let
    indexedLayers :: Vec NumLayers (Index NumLayers, TransformerLayerComponent, Cache.KVRamOwner dom)
    indexedLayers = imap (\i (comp, owner) -> (i, comp, owner)) (zip transformerLayers cacheOwners)

    (finalLayerData, doneFlags) = mapAccumL (layerProcessor processingState) initLayerData indexedLayers

  in
    (finalLayerData, doneFlags)

layerProcessor :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom LayerData
  -> (Index NumLayers, TransformerLayerComponent, Cache.KVRamOwner dom)
  -> (Signal dom LayerData, (Signal dom Bool, Signal dom Bool))
layerProcessor processingState currentLayerData (layerIndex, layerComp, cacheOwner) =
  let
    (newLayerData, writeDone, attnDone, commitC3) =
      TransformerLayer.multiCycleTransformerLayer layerComp cacheOwner layerIndex processingState currentLayerData

    selectedLayerData =
      liftA4 (layerDataSelector layerIndex) processingState currentLayerData newLayerData commitC3

  in
    (selectedLayerData, (writeDone, attnDone))

layerDataSelector :: Index NumLayers
                  -> ProcessingState
                  -> LayerData
                  -> LayerData
                  -> LayerData
                  -> LayerData
layerDataSelector layerIndex state currentData newData stage3Data =
  let
    curLayer = processingLayer state
    curStage = processingStage state
  in
    case () of
      _ | curLayer /= layerIndex       -> currentData
        | curStage == Stage3_Attend    -> stage3Data
        | otherwise                    -> newData
