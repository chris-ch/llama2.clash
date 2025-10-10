module LLaMa2.Core.Transformer (
    multiCycleTransformer
) where

import Clash.Prelude
import LLaMa2.Core.Transformer.Internal
import LLaMa2.Helpers (liftA4)
import LLaMa2.Core.Types
  ( LayerData(..)
  , ProcessingState (..)
  , CycleStage (..)
  , Temperature, Seed
  , Token
  )
import LLaMa2.Config
  (  NumLayers, VocabularySize, LLaMa2Dimension
  )
import qualified LLaMa2.Layers.TransformerLayer as TransformerLayer
  ( TransformerDecoderComponent(..)
  , multiCycleTransformerLayer
  )
import LLaMa2.Layers.TransformerLayer (TransformerDecoderComponent(..), TransformerLayerComponent (..))
import qualified LLaMa2.Embedding.PRNG as PRNG (tokenSampler)
import qualified LLaMa2.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs (..)
  )
import qualified LLaMa2.Core.Embedding as Embedding (embedder)
import qualified LLaMa2.Layers.Components.Quantized as Quantized (EmbeddingComponentQ(..))
import LLaMa2.Numeric.ParamPack (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint)

type LayerProcessorData dom = (Signal dom LayerData, Vec NumLayers (Signal dom Bool, Signal dom Bool))

multiCycleTransformer :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom Token
  -> Signal dom Bool            -- ^ inputTokenValid (True while external prompt is used)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token , Signal dom Bool )
multiCycleTransformer decoder inputToken inputTokenValid temperature seed =
  ( selectedToken, readyPulse)
 where
  vocabulary :: MatI8E VocabularySize LLaMa2Dimension
  vocabulary = Quantized.vocabularyQ $ modelEmbedding decoder

  transformerLayers :: Vec NumLayers TransformerLayerComponent
  transformerLayers  = modelLayers decoder

  -- Select this layer's s1Done for pipeline control
  s1DoneThisLayer :: Signal dom Bool
  s1DoneThisLayer = pure True

  pipelineController :: PipelineController.PipelineOutputs dom
  pipelineController =
    PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer s1DoneThisLayer inputTokenValid

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

  -- Quantized embedding lookup via ROM (1-cycle)
  tokenEmbedding :: Signal dom (Vec LLaMa2Dimension FixedPoint)
  tokenEmbedding = Embedding.embedder vocabulary selectedToken

  layerDataRegister :: Signal dom LayerData
  layerDataRegister = register initialLayerData nextLayerData

  layerInputSelector :: ProcessingState -> LayerData -> Vec LLaMa2Dimension FixedPoint -> LayerData
  layerInputSelector ps currentLayerData tokenEmbed
    | processingStage ps /= Stage1_ProjectQKV = currentLayerData
    | processingLayer ps == 0                 = currentLayerData { inputVector = tokenEmbed }
    | otherwise                               = currentLayerData { inputVector = feedForwardOutput currentLayerData }

  selectedInput :: Signal dom LayerData
  selectedInput = liftA3 layerInputSelector processingState layerDataRegister tokenEmbedding

  (nextLayerData, doneFlags) = pipelineProcessor processingState selectedInput transformerLayers

  (writeDone, attnDone) = unzip doneFlags

  writeDoneThisLayer :: Signal dom Bool
  writeDoneThisLayer = (!!) <$> sequenceA writeDone <*> layerIndex

  attnDoneThisLayer :: Signal dom Bool
  attnDoneThisLayer  = (!!) <$> sequenceA attnDone <*> layerIndex

pipelineProcessor :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom LayerData
  -> Vec NumLayers TransformerLayerComponent
  -> LayerProcessorData dom
pipelineProcessor processingState initLayerData transformerLayers =
  let
    indexedLayers = imap (,) transformerLayers

    (finalLayerData, doneFlags) = mapAccumL (layerProcessor processingState) initLayerData indexedLayers

  in
    (finalLayerData, doneFlags)

layerProcessor :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom LayerData
  -> (Index NumLayers, TransformerLayerComponent)
  -> (Signal dom LayerData, (Signal dom Bool, Signal dom Bool))
layerProcessor processingState currentLayerData (layerIndex, layerComp) =
  let
    (newLayerData, writeDone, attnDone, commitC3) =
      TransformerLayer.multiCycleTransformerLayer layerComp layerIndex processingState currentLayerData

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
