-- | LLaMa2 Decoder - Top-level orchestration
-- Simplified to focus on: token flow, embedding, output projection, and sampling
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
  ( NumLayers, VocabularySize, ModelDimension, SequenceLength
  , NumQueryHeads, NumKeyValueHeads, HeadDimension
  )
import LLaMa2.Numeric.ParamPack (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint)

-- Import sub-modules
import qualified LLaMa2.Embedding.PRNG as PRNG (tokenSampler)
import qualified LLaMa2.Core.Embedding as Embedding (embedder)
import qualified LLaMa2.Layer.Components.Quantized as Quantized (EmbeddingComponentQ(..))
import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (logitsProjector)
import qualified LLaMa2.Decoder.SequenceController as SequenceController 
  ( PipelineOutputs (..), layerSequencer, pipelineController, processingState )
import qualified LLaMa2.Decoder.LayerStack as LayerStack (processLayers)

-- ============================================================================
-- Initial State
-- ============================================================================

initialLayerData :: LayerData
initialLayerData = LayerData
  { inputVector       = repeat 0          :: Vec ModelDimension FixedPoint
  , queryVectors      = repeat (repeat 0) :: Vec NumQueryHeads (Vec HeadDimension FixedPoint)
  , keyVectors        = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , valueVectors      = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , attentionOutput   = repeat 0          :: Vec ModelDimension FixedPoint
  , feedForwardOutput = repeat 0          :: Vec ModelDimension FixedPoint
  }

-- ============================================================================
-- Introspection
-- ============================================================================

-- | Introspection signals for runtime visibility
data DecoderIntrospection dom = DecoderIntrospection
  { state         :: Signal dom ProcessingState
  , logitsValid   :: Signal dom Bool
  , attnDone      :: Signal dom Bool
  , qkvDone       :: Signal dom Bool
  , ffnDone       :: Signal dom Bool
  , writeDone     :: Signal dom Bool
  , inputToken    :: Signal dom Token
  , outputToken   :: Signal dom Token
  , feedbackToken :: Signal dom Token
  , embeddingNorm :: Signal dom FixedPoint
  , outputNorm    :: Signal dom FixedPoint
  , layerIndex    :: Signal dom (Index NumLayers)
  , seqPos        :: Signal dom (Index SequenceLength)
  , ready         :: Signal dom Bool
  } deriving (Generic, NFDataX)

-- ============================================================================
-- Main Decoder
-- ============================================================================

decoder :: forall dom
   . HiddenClockResetEnable dom
  => DecoderParameters
  -> Signal dom Token                    -- ^ Input token
  -> Signal dom Bool                     -- ^ Input token valid
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token                  -- ^ Output token
     , Signal dom Bool                   -- ^ Ready pulse
     , DecoderIntrospection dom          -- ^ Introspection signals
     )
decoder params inputToken inputTokenValid temperature seed =
  (outputToken, readyPulse, introspection)
  where
    -- ========================================================================
    -- 1. SEQUENCE CONTROL
    -- ========================================================================
    -- Track which layer and sequence position we're processing
    
    (layerIdx, seqPosIdx, readyPulse) =
      SequenceController.layerSequencer ffnDoneThisLayer
    
    -- Stage controller manages internal layer stages (QKV, Write, Attn, FFN)
    pipelineCtrl = SequenceController.pipelineController
      attnDoneThisLayer
      writeDoneThisLayer
      qkvDoneThisLayer
      ffnDoneThisLayer
      logitsValid
      inputTokenValid
    
    processingState = SequenceController.processingState pipelineCtrl
    
    -- ========================================================================
    -- 2. TOKEN SELECTION & EMBEDDING
    -- ========================================================================
    -- Choose between external input or generated feedback token
    
    outputToken = mux inputTokenValid inputToken feedbackToken
    
    -- Lookup token embedding from vocabulary
    vocabulary :: MatI8E VocabularySize ModelDimension
    vocabulary = Quantized.vocabularyQ $ modelEmbedding params
    
    tokenEmbedding :: Signal dom (Vec ModelDimension FixedPoint)
    tokenEmbedding = Embedding.embedder vocabulary outputToken
    
    embeddingNorm = sum . map abs <$> tokenEmbedding
    
    -- ========================================================================
    -- 3. LAYER STACK PROCESSING
    -- ========================================================================
    -- Process through all transformer layers
    
    -- Maintain layer data state
    layerDataReg :: Signal dom LayerData
    layerDataReg = register initialLayerData nextLayerData
    
    -- Select input for current layer:
    -- - Layer 0 gets token embedding
    -- - Other layers get previous layer's FFN output
    layerInput :: Signal dom LayerData
    layerInput = prepareLayerInput <$> layerIdx <*> layerDataReg <*> tokenEmbedding
    
    prepareLayerInput :: Index NumLayers -> LayerData -> Vec ModelDimension FixedPoint -> LayerData
    prepareLayerInput idx currentData embedding
      | idx == 0  = currentData { inputVector = embedding }
      | otherwise = currentData { inputVector = feedForwardOutput currentData }
    
    -- Process all layers (only active layer computes)
    transformerLayers = modelLayers params
    
    (nextLayerData, doneFlags) = 
      LayerStack.processLayers 
        processingState 
        layerIdx 
        layerInput 
        transformerLayers
    
    -- Extract completion flags
    (writeDoneVec, attnDoneVec, qkvDoneVec, _qkvReadyVec, ffnDoneVec) = unzip5 doneFlags
    
    -- Select flags for current layer
    writeDoneThisLayer = selectCurrent layerIdx writeDoneVec
    attnDoneThisLayer  = selectCurrent layerIdx attnDoneVec
    qkvDoneThisLayer   = selectCurrent layerIdx qkvDoneVec
    ffnDoneThisLayer   = selectCurrent layerIdx ffnDoneVec
    
    -- ========================================================================
    -- 4. OUTPUT PROJECTION & SAMPLING
    -- ========================================================================
    -- After last layer completes, project to vocabulary and sample
    
    lastLayerFfnDone = (layerIdx .==. pure maxBound) .&&. ffnDoneThisLayer
    
    layerOutput = feedForwardOutput <$> nextLayerData
    outputNorm = sum . map abs <$> layerOutput
    
    -- Project final layer output to logits
    (logits, logitsValid) =
      OutputProjection.logitsProjector 
        lastLayerFfnDone 
        (pure True) 
        params 
        layerOutput
    
    -- Sample next token from logits
    sampledToken = PRNG.tokenSampler logitsValid temperature seed logits
    
    -- Register sampled token for feedback
    feedbackToken = regEn 0 logitsValid sampledToken
    
    -- ========================================================================
    -- 5. INTROSPECTION
    -- ========================================================================
    
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
      , layerIndex    = layerIdx
      , seqPos        = seqPosIdx
      , ready         = readyPulse
      }

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- | Select completion flag for the currently active layer
selectCurrent
  :: Signal dom (Index NumLayers)
  -> Vec NumLayers (Signal dom Bool)
  -> Signal dom Bool
selectCurrent idx flags = (!!) <$> sequenceA flags <*> idx
