module LLaMa2.Decoder.Decoder (
    decoder, DecoderIntrospection(..)
) where
import Clash.Prelude

import LLaMa2.Core.Types
  ( LayerData(..)
  , ProcessingState (..)
  , Temperature, Seed
  , Token
  )

import LLaMa2.Types.Parameters (DecoderParameters (..))
import LLaMa2.Config
  (  NumLayers, VocabularySize, ModelDimension, SequenceLength, NumQueryHeads, NumKeyValueHeads, HeadDimension
  )
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer
  ( transformerLayer
  )
import LLaMa2.Layer.TransformerLayer (TransformerLayerComponent (..))
import qualified LLaMa2.Embedding.PRNG as PRNG (tokenSampler)

import qualified LLaMa2.Core.Embedding as Embedding (embedder)
import qualified LLaMa2.Layer.Components.Quantized as Quantized (EmbeddingComponentQ(..))
import LLaMa2.Numeric.ParamPack (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (logitsProjector)
import qualified LLaMa2.Decoder.SequenceController as SequenceController (PipelineOutputs (..), layerSequencer, pipelineController)

initialLayerData :: LayerData
initialLayerData = LayerData
  { inputVector       = repeat 0          :: Vec ModelDimension FixedPoint
  , queryVectors      = repeat (repeat 0) :: Vec NumQueryHeads (Vec HeadDimension FixedPoint)
  , keyVectors        = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , valueVectors      = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , attentionOutput   = repeat 0          :: Vec ModelDimension FixedPoint
  , feedForwardOutput = repeat 0          :: Vec ModelDimension FixedPoint
  }

-- | Introspection signals — meant for runtime visibility / observability
data DecoderIntrospection dom = DecoderIntrospection
  { state         :: Signal dom ProcessingState
  , logitsValid   :: Signal dom Bool
  , attnDone      :: Signal dom Bool
  , qkvDone       :: Signal dom Bool
  , ffnDone       :: Signal dom Bool
  , writeDone     :: Signal dom Bool
  , inputToken    :: Signal dom Token
  , outputToken :: Signal dom Token
  , feedbackToken :: Signal dom Token
  , embeddingNorm :: Signal dom FixedPoint
  , outputNorm    :: Signal dom FixedPoint
  , layerIndex :: Signal dom (Index NumLayers)
  , seqPos     :: Signal dom (Index SequenceLength)
  , ready      :: Signal dom Bool
  } deriving (Generic, NFDataX)

decoder :: forall dom
   . HiddenClockResetEnable dom
  => DecoderParameters
  -> Signal dom Token
  -> Signal dom Bool            -- ^ inputTokenValid (True while external prompt is used)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
      , Signal dom Bool
      , DecoderIntrospection dom -- ^ introspection signals
     )
decoder params inputToken inputTokenValid temperature seed =
  ( outputToken, readyPulse, introspection)
 where
  transformerLayers :: Vec NumLayers TransformerLayerComponent
  transformerLayers  = modelLayers params
  
  (newLayerIdx, newSeqPosIdx, readyPulse) =
    SequenceController.layerSequencer ffnDoneThisLayer

  qkvDoneThisLayer :: Signal dom Bool
  qkvDoneThisLayer = (!!) <$> sequenceA qkvDone <*> newLayerIdx

  ffnDoneThisLayer :: Signal dom Bool
  ffnDoneThisLayer = (!!) <$> sequenceA ffnDone <*> newLayerIdx

  -- STAGE CONTROLLER - Provides stage sequencing for layer internals
  -- Note: Top-level layer tracking uses minimalController (newLayerIdx)

  processingState :: Signal dom ProcessingState
  processingState = SequenceController.processingState (
    SequenceController.pipelineController
      attnDoneThisLayer
      writeDoneThisLayer
      qkvDoneThisLayer
      ffnDoneThisLayer
      logitsValid
      inputTokenValid)

  vocabulary :: MatI8E VocabularySize ModelDimension
  vocabulary = Quantized.vocabularyQ $ modelEmbedding params

  -- Quantized embedding lookup via ROM (1-cycle)
  tokenEmbedding :: Signal dom (Vec ModelDimension FixedPoint)
  tokenEmbedding = Embedding.embedder vocabulary outputToken

  layerDataRegister :: Signal dom LayerData
  layerDataRegister = register initialLayerData nextLayerData

  -- Always prepare correct input based on layer
  -- The layer itself will use inputVector when it needs it (at Stage1)
  layerInputSelector :: Index NumLayers -> LayerData -> Vec ModelDimension FixedPoint -> LayerData
  layerInputSelector newIdx currentLayerData tokenEmbed
    | newIdx == 0   = currentLayerData { inputVector = tokenEmbed }
    | otherwise     = currentLayerData { inputVector = feedForwardOutput currentLayerData }

  selectedInput :: Signal dom LayerData
  selectedInput = layerInputSelector <$> newLayerIdx <*> layerDataRegister <*> tokenEmbedding

  -- Pass newLayerIdx to pipelineProcessor
  (nextLayerData, doneFlags) =
    pipelineProcessor processingState newLayerIdx selectedInput transformerLayers

  -- Unpack the completion signals
  (writeDone, attnDone, qkvDone, _qkvReady, ffnDone) = unzip5 doneFlags

  writeDoneThisLayer :: Signal dom Bool
  writeDoneThisLayer = (!!) <$> sequenceA writeDone <*> newLayerIdx

  attnDoneThisLayer :: Signal dom Bool
  attnDoneThisLayer  = (!!) <$> sequenceA attnDone <*> newLayerIdx

  -- Sequential classifier starts when last layer FFN completes
  -- ffnDoneThisLayer already implies Stage4
  lastLayerFfnDone = 
    (newLayerIdx .==. pure maxBound) .&&. 
    ffnDoneThisLayer

  -- Extract final layer output (updated by ffnValidOut at Stage4)
  layerOutput :: Signal dom (Vec ModelDimension FixedPoint)
  layerOutput = feedForwardOutput <$> nextLayerData

  (logits, logitsValid) =
    OutputProjection.logitsProjector lastLayerFfnDone (pure True) params layerOutput

  outputNorm :: Signal dom FixedPoint
  outputNorm = sum . map abs <$> layerOutput

  -- Token sampling from logits 
  sampledToken :: Signal dom Token
  sampledToken = PRNG.tokenSampler logitsValid temperature seed logits

  -- Feedback register
  feedbackToken :: Signal dom Token
  feedbackToken = regEn 0 logitsValid sampledToken

  outputToken :: Signal dom Token
  outputToken = mux inputTokenValid inputToken feedbackToken

  -- Lightweight vector diagnostics (sum of abs values)
  embeddingNorm :: Signal dom FixedPoint
  embeddingNorm = sum . map abs <$> tokenEmbedding

  introspection :: DecoderIntrospection dom
  introspection = DecoderIntrospection
    { state         = processingState
    , logitsValid
    , attnDone      = attnDoneThisLayer
    , qkvDone       = qkvDoneThisLayer
    , ffnDone       = ffnDoneThisLayer
    , writeDone     = writeDoneThisLayer
    , inputToken
    , outputToken
    , feedbackToken
    , embeddingNorm
    , outputNorm
    , layerIndex = newLayerIdx
    , seqPos     = newSeqPosIdx
    , ready      = readyPulse
    }

-- | Process all layers sequentially in the pipeline
pipelineProcessor
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom (Index NumLayers)
  -> Signal dom LayerData
  -> Vec NumLayers TransformerLayerComponent
  -> ( Signal dom LayerData
     , Vec NumLayers (Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool)
     )
pipelineProcessor psSig currentLayerSig initLayerData layers =
  let
    -- Process each layer sequentially, accumulating only the layer data
    (finalData, layerResults) = 
      mapAccumL (\ld (ix, comp) -> 
        let (ldNext, doneFlags, vOut, rOut) = layerProcessor psSig currentLayerSig ld (ix, comp)
        in (ldNext, (doneFlags, vOut, rOut))
      ) initLayerData (imap (,) layers)

    -- Extract required outputs
    doneFlagsVec = fmap (\(df, _, _) -> df) layerResults
  in
    (finalData, doneFlagsVec)

-- | Process a single transformer layer
layerProcessor :: HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom (Index NumLayers)            -- ^ NEW: currentLayer from new controller
  -> Signal dom LayerData                    -- ^ input data
  -> (Index NumLayers, TransformerLayerComponent)
  -> ( Signal dom LayerData
     , (Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool)
     , Signal dom Bool                     -- ^ validOut (FFN done)
     , Signal dom Bool                     -- ^ readyIn
     )
layerProcessor psSig currentLayerSig ldIn (layerIndex, layerComp) =
  let
    -- Compute layerActive signal: true when this is the current layer
    layerActive = currentLayerSig .==. pure layerIndex
    
    -- Call transformerLayer with NEW layerActive parameter
    ( ldNext
      , writeDone
      , attnDone
      , qkvDone
      , ldAfterAttn
      , qkvInReady
      , ffnDone
      ) = TransformerLayer.transformerLayer layerComp layerIndex psSig layerActive ldIn

    -- Ready/Valid handshake
    validOut = ffnDone

    selectedLd = mux (currentLayerSig .==. pure layerIndex)
                     ldNext
                     ldIn
  in
    (selectedLd, (writeDone, attnDone, qkvDone, qkvInReady, ffnDone), validOut, pure True)
