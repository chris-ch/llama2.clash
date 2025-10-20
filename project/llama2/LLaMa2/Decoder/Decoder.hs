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
  ( NumLayers, ModelDimension
  , NumQueryHeads, NumKeyValueHeads, HeadDimension
  )
import LLaMa2.Numeric.Types (FixedPoint)

-- Import sub-modules
import qualified LLaMa2.Embedding.PRNG as PRNG (tokenSampler)
import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (outputProjection)
import qualified LLaMa2.Decoder.SequenceController as SequenceController
  ( PipelineOutputs (..), pipelineController, processingState, SequenceState (..), sequenceController )
import qualified LLaMa2.Decoder.LayerStack as LayerStack (processLayers, getCurrentLayerFlag)
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding

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
  , layerIndex    :: Signal dom (Index NumLayers)
  , ready         :: Signal dom Bool
  , attnDone      :: Signal dom Bool
  , ffnDone       :: Signal dom Bool
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
    -- SEQUENCE CONTROL
    -- ========================================================================
    
    (seqState, readyPulse) = SequenceController.sequenceController ffnDoneThisLayer
    
    -- Extract layer index and sequence position
    layerIdx = SequenceController.currentLayer <$> seqState
    
    -- Stage controller manages internal layer stages (QKV, Write, Attn, FFN)
    processingState = SequenceController.processingState (
      SequenceController.pipelineController
        attnDoneThisLayer
        writeDoneThisLayer
        qkvDoneThisLayer
        ffnDoneThisLayer
        logitsValid
        inputTokenValid)
    
    -- ========================================================================
    -- TOKEN SELECTION & EMBEDDING
    -- ========================================================================
      
    -- Token selection: external input or feedback
    outputToken = mux inputTokenValid inputToken feedbackToken
    
    -- Lookup token embedding from vocabulary
    embeddedVector :: Signal dom (Vec ModelDimension FixedPoint)
    embeddedVector = InputEmbedding.inputEmbedding (modelEmbedding params) outputToken

    -- ========================================================================
    -- LAYER STACK PROCESSING
    -- ========================================================================
    
    -- Maintain layer data state
    layerDataReg :: Signal dom LayerData
    layerDataReg = register initialLayerData nextLayerData
    
    -- Select input for current layer:
    -- - Layer 0 gets token embedding
    -- - Other layers get previous layer's FFN output
    layerInput :: Signal dom LayerData
    layerInput = prepareLayerInput <$> layerIdx <*> layerDataReg <*> embeddedVector
    
    prepareLayerInput :: Index NumLayers -> LayerData -> Vec ModelDimension FixedPoint -> LayerData
    prepareLayerInput idx currentData embedding
      | idx == 0  = currentData { inputVector = embedding }
      | otherwise = currentData { inputVector = feedForwardOutput currentData }
    
    -- Process all layers (only active layer computes)
    (nextLayerData, doneFlags) = 
      LayerStack.processLayers 
        processingState 
        layerIdx 
        layerInput 
        (modelLayers params)
    
    -- Extract current layer completion flags
    (writeDone, attnDone, qkvDone, _qkvReady, ffnDone) = unzip5 doneFlags
    writeDoneThisLayer = LayerStack.getCurrentLayerFlag layerIdx writeDone
    attnDoneThisLayer  = LayerStack.getCurrentLayerFlag layerIdx attnDone
    qkvDoneThisLayer   = LayerStack.getCurrentLayerFlag layerIdx qkvDone
    ffnDoneThisLayer   = LayerStack.getCurrentLayerFlag layerIdx ffnDone
    
    -- ========================================================================
    -- OUTPUT PROJECTION & SAMPLING
    -- ========================================================================
    
    lastLayerFfnDone = (layerIdx .==. pure maxBound) .&&. ffnDoneThisLayer
    
    layerOutput = feedForwardOutput <$> nextLayerData

    -- Output projection (unembedding + logits)
    (logits, logitsValid) =
      OutputProjection.outputProjection 
        params 
        lastLayerFfnDone 
        layerOutput
    
    -- Sample next token from logits
    sampledToken = PRNG.tokenSampler logitsValid temperature seed logits
    
    -- Register sampled token for feedback
    feedbackToken = regEn 0 logitsValid sampledToken
    
    -- ========================================================================
    -- INTROSPECTION
    -- ========================================================================
    
    introspection = DecoderIntrospection
      { state         = processingState
      , attnDone      = attnDoneThisLayer
      , ffnDone       = ffnDoneThisLayer
      , layerIndex    = layerIdx
      , ready         = readyPulse
      }
