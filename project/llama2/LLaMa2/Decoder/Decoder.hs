-- | LLaMa2 Decoder - Top-level orchestration
-- Simplified to focus on: token flow, embedding, output projection, and sampling
module LLaMa2.Decoder.Decoder (
    decoder, DecoderIntrospection(..)
) where

import Clash.Prelude

import LLaMa2.Types.LayerData
  ( LayerData(..)
  , ProcessingState (..)
  , Temperature, Seed
  , Token
  )
import LLaMa2.Types.Parameters (DecoderParameters (..))
import LLaMa2.Types.ModelConfig 
  ( NumLayers, ModelDimension
  , NumQueryHeads, NumKeyValueHeads, HeadDimension
  )
import LLaMa2.Numeric.Types (FixedPoint)

-- Import sub-modules
import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (outputProjection)
import qualified LLaMa2.Decoder.SequenceController as SequenceController
  ( pipelineController, processingState, sequenceController, SequenceState (..) )
import qualified LLaMa2.Decoder.LayerStack as LayerStack (processLayers, getCurrentLayerFlag, prepareLayerInput)
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding
import qualified LLaMa2.Sampling.Sampler as Sampler

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
    
    layerIdx :: Signal dom (Index NumLayers)
    layerIdx = SequenceController.currentLayer <$> seqState
    
    processingState :: Signal dom ProcessingState
    processingState = SequenceController.processingState (
      SequenceController.pipelineController
        attnDoneThisLayer writeDoneThisLayer qkvDoneThisLayer
        ffnDoneThisLayer logitsValid inputTokenValid)
    
    -- ========================================================================
    -- TOKEN SELECTION & EMBEDDING
    -- ========================================================================
    
    outputToken :: Signal dom Token
    outputToken = mux inputTokenValid inputToken feedbackToken
    
    embeddedVector :: Signal dom (Vec ModelDimension FixedPoint)
    embeddedVector = InputEmbedding.inputEmbedding (modelEmbedding params) outputToken

    -- ========================================================================
    -- LAYER STACK PROCESSING
    -- ========================================================================
    
    layerInput :: Signal dom LayerData
    layerInput = LayerStack.prepareLayerInput <$> layerIdx 
                   <*> register initialLayerData nextLayerData 
                   <*> embeddedVector

    (nextLayerData, doneFlags) = 
      LayerStack.processLayers processingState layerIdx layerInput (modelLayers params)
    
    (writeDone, attnDone, qkvDone, _, ffnDone) = unzip5 doneFlags
    writeDoneThisLayer = LayerStack.getCurrentLayerFlag layerIdx writeDone
    attnDoneThisLayer  = LayerStack.getCurrentLayerFlag layerIdx attnDone
    qkvDoneThisLayer   = LayerStack.getCurrentLayerFlag layerIdx qkvDone
    ffnDoneThisLayer   = LayerStack.getCurrentLayerFlag layerIdx ffnDone
    
    -- ========================================================================
    -- OUTPUT PROJECTION & SAMPLING
    -- ========================================================================
    
    layerOutput :: Signal dom (Vec ModelDimension FixedPoint)
    layerOutput = feedForwardOutput <$> nextLayerData
    
    lastLayerComplete :: Signal dom Bool
    lastLayerComplete = (layerIdx .==. pure maxBound) .&&. ffnDoneThisLayer

    (logits, logitsValid) = OutputProjection.outputProjection params lastLayerComplete layerOutput
    
    sampledToken :: Signal dom Token
    sampledToken = Sampler.tokenSampler logitsValid temperature seed logits
    
    feedbackToken :: Signal dom Token
    feedbackToken = regEn 0 logitsValid sampledToken
    
    -- ========================================================================
    -- INTROSPECTION
    -- ========================================================================
    
    introspection = DecoderIntrospection
      { state = processingState, layerIndex = layerIdx, ready = readyPulse
      , attnDone = attnDoneThisLayer, ffnDone = ffnDoneThisLayer }
