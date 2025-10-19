module LLaMa2.Core.Transformer (
    transformer, TransformerIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Core.Transformer.Internal
import LLaMa2.Core.Types
  ( LayerData(..)
  , ProcessingState (..)
  , CycleStage (..)
  , Temperature, Seed
  , Token
  )
import LLaMa2.Config
  (  NumLayers, VocabularySize, ModelDimension, SequenceLength
  )
import qualified LLaMa2.Layers.TransformerLayer as TransformerLayer
  ( TransformerDecoderComponent(..)
  , transformerLayer
  )
import LLaMa2.Layers.TransformerLayer (TransformerDecoderComponent(..), TransformerLayerComponent (..))
import qualified LLaMa2.Embedding.PRNG as PRNG (tokenSamplerFromLogits)
import qualified LLaMa2.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs (..)
  , runMinimalController  -- NEW
  , ControllerState(..)   -- NEW
  )
import qualified LLaMa2.Core.Embedding as Embedding (embedder)
import qualified LLaMa2.Layers.Components.Quantized as Quantized (EmbeddingComponentQ(..))
import LLaMa2.Numeric.ParamPack (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Embedding.PRNG (transformerLogitsSeq)

-- | Introspection signals — meant for runtime visibility / observability
data TransformerIntrospection dom = TransformerIntrospection
  { state         :: Signal dom ProcessingState
  , layerIndex    :: Signal dom (Index NumLayers)
  , ready         :: Signal dom Bool
  , logitsValid   :: Signal dom Bool
  , attnDone      :: Signal dom Bool
  , qkvDone       :: Signal dom Bool
  , ffnDone       :: Signal dom Bool
  , writeDone     :: Signal dom Bool
  , inputToken    :: Signal dom Token
  , selectedToken :: Signal dom Token
  , feedbackToken :: Signal dom Token
  , embeddingNorm :: Signal dom FixedPoint
  , outputNorm    :: Signal dom FixedPoint
  -- NEW: signals from new controller
  , newLayerIndex :: Signal dom (Index NumLayers)
  , newSeqPos     :: Signal dom (Index SequenceLength)
  , newReady      :: Signal dom Bool
  } deriving (Generic, NFDataX)

transformer :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom Token
  -> Signal dom Bool            -- ^ inputTokenValid (True while external prompt is used)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
      , Signal dom Bool
      , TransformerIntrospection dom -- ^ introspection signals
     )
transformer decoder inputToken inputTokenValid temperature seed =
  ( selectedToken, readyPulse, introspection)
 where
  vocabulary :: MatI8E VocabularySize ModelDimension
  vocabulary = Quantized.vocabularyQ $ modelEmbedding decoder

  transformerLayers :: Vec NumLayers TransformerLayerComponent
  transformerLayers  = modelLayers decoder
  
  downstreamReady :: Signal dom Bool
  downstreamReady = pure True  -- always ready
  
  -- OLD CONTROLLER (still driving everything)
  pipelineController =
    PipelineController.runPipelineController
      attnDoneThisLayer
      writeDoneThisLayer
      qkvDoneThisLayer
      ffnDoneThisLayer
      logitsValid
      inputTokenValid
      downstreamReady

  -- Sequential classifier starts when last layer FFN completes
  lastLayerFfnDone = 
    ((processingStage <$> processingState) .==. pure Stage4_FeedForward) .&&.
    ((processingLayer <$> processingState) .==. pure maxBound) .&&.
    ffnDoneThisLayer

  readyPulse :: Signal dom Bool
  readyPulse = newReadyPulse

  processingState :: Signal dom ProcessingState
  processingState = PipelineController.processingState pipelineController

  -- NEW CONTROLLER (running in parallel)
  -- MIGRATED: Removed stage check - ffnDoneThisLayer already implies Stage4 completion
  currentLayerDone = ffnDoneThisLayer
  
  (newLayerIdx, newSeqPosIdx, newReadyPulse) =
    PipelineController.runMinimalController currentLayerDone inputTokenValid

  -- Extract final layer output (updated by ffnValidOut at Stage4)
  finalLayerOutput :: Signal dom (Vec ModelDimension FixedPoint)
  finalLayerOutput = feedForwardOutput <$> nextLayerData

  (logits, logitsValid, _) =
    transformerLogitsSeq lastLayerFfnDone (pure True) decoder finalLayerOutput

  -- keep 
  tokenSampleSeq :: Signal dom Token
  tokenSampleSeq = PRNG.tokenSamplerFromLogits logitsValid temperature seed logits

  -- Register token when logits become valid (not just readyPulse)
  feedbackToken :: Signal dom Token
  feedbackToken = regEn 0 logitsValid tokenSampleSeq

  selectedToken :: Signal dom Token
  selectedToken = mux inputTokenValid inputToken feedbackToken

  -- Quantized embedding lookup via ROM (1-cycle)
  tokenEmbedding :: Signal dom (Vec ModelDimension FixedPoint)
  tokenEmbedding = Embedding.embedder vocabulary selectedToken

  layerDataRegister :: Signal dom LayerData
  layerDataRegister = register initialLayerData nextLayerData

  -- MIGRATED: Removed stage check - always prepare correct input based on layer
  -- The layer itself will use inputVector when it needs it (at Stage1)
  layerInputSelector :: Index NumLayers -> LayerData -> Vec ModelDimension FixedPoint -> LayerData
  layerInputSelector newIdx currentLayerData tokenEmbed
    | newIdx == 0   = currentLayerData { inputVector = tokenEmbed }
    | otherwise     = currentLayerData { inputVector = feedForwardOutput currentLayerData }

  selectedInput :: Signal dom LayerData
  selectedInput = layerInputSelector <$> newLayerIdx <*> layerDataRegister <*> tokenEmbedding

  (nextLayerData, _finalValidOut, doneFlags) =
    pipelineProcessor (pure True) (pure True) processingState selectedInput transformerLayers

  -- Unpack the completion signals
  (writeDone, attnDone, qkvDone, _qkvReady, ffnDone) = unzip5 doneFlags

  -- MIGRATED: All uses now reference newLayerIdx from new controller
  writeDoneThisLayer :: Signal dom Bool
  writeDoneThisLayer = (!!) <$> sequenceA writeDone <*> newLayerIdx

  attnDoneThisLayer :: Signal dom Bool
  attnDoneThisLayer  = (!!) <$> sequenceA attnDone <*> newLayerIdx

  qkvDoneThisLayer :: Signal dom Bool
  qkvDoneThisLayer = (!!) <$> sequenceA qkvDone <*> newLayerIdx

  ffnDoneThisLayer :: Signal dom Bool
  ffnDoneThisLayer = (!!) <$> sequenceA ffnDone <*> newLayerIdx

  -- Lightweight vector diagnostics (sum of abs values)
  embeddingNorm :: Signal dom FixedPoint
  embeddingNorm = sum . map abs <$> tokenEmbedding

  outputNorm :: Signal dom FixedPoint
  outputNorm = sum . map abs <$> finalLayerOutput

  introspection :: TransformerIntrospection dom
  introspection = TransformerIntrospection
    { state         = processingState
    , layerIndex    = newLayerIdx  -- MIGRATED: now uses new controller
    , ready         = readyPulse
    , logitsValid
    , attnDone      = attnDoneThisLayer
    , qkvDone       = qkvDoneThisLayer
    , ffnDone       = ffnDoneThisLayer
    , writeDone     = writeDoneThisLayer
    , inputToken
    , selectedToken
    , feedbackToken
    , embeddingNorm
    , outputNorm
    -- NEW: expose new controller signals
    , newLayerIndex = newLayerIdx
    , newSeqPos     = newSeqPosIdx
    , newReady      = newReadyPulse
    }

-- | Process all layers sequentially in the pipeline
pipelineProcessor
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool                        -- ^ validIn (input token ready)
  -> Signal dom Bool                        -- ^ readyOut (next stage ready)
  -> Signal dom ProcessingState
  -> Signal dom LayerData                   -- ^ current layer input
  -> Vec NumLayers TransformerLayerComponent
  -> ( Signal dom LayerData
     , Signal dom Bool                        -- ^ validOut of last layer
     , Vec NumLayers (Signal dom Bool         -- writeDone
                     , Signal dom Bool        -- attnDone
                     , Signal dom Bool        -- qkvDone
                     , Signal dom Bool        -- qkvInReady
                     , Signal dom Bool)       -- ffnDone
     )
pipelineProcessor validIn readyOut psSig initLayerData layers =
  let
    indexedLayers = imap (,) layers

    stepLayer (ld, vIn) (ix, comp) =
      let
        (ldNext, doneFlags, vOut, rOut) =
          layerProcessor psSig vIn readyOut ld (ix, comp)
      in ((ldNext, vOut), (doneFlags, vOut, rOut))

    (finalState, layerResults) = mapAccumL stepLayer (initLayerData, validIn) indexedLayers

    doneFlagsVec = map (\(df, _, _) -> df) layerResults
    validOuts    = map (\(_, vOut, _) -> vOut) layerResults

    finalValidOut = last validOuts
  in
    (fst finalState, finalValidOut, doneFlagsVec)

-- | Process a single transformer layer
layerProcessor :: HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom Bool                     -- ^ validIn
  -> Signal dom Bool                     -- ^ readyOut from next layer
  -> Signal dom LayerData                -- ^ input data
  -> (Index NumLayers, TransformerLayerComponent)
  -> ( Signal dom LayerData
     , (Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool, Signal dom Bool)
     , Signal dom Bool                     -- ^ validOut (FFN done)
     , Signal dom Bool                     -- ^ readyIn
     )
layerProcessor psSig validIn readyOut ldIn (layerIndex, layerComp) =
  let
    -- Call transformerLayer with the REAL signature (7 outputs)
    ( ldNext
      , writeDone
      , attnDone
      , qkvDone
      , ldAfterAttn
      , qkvInReady
      , ffnDone
      ) = TransformerLayer.transformerLayer layerComp layerIndex psSig ldIn

    -- Ready/Valid handshake
    validOut = ffnDone
    readyIn  = readyOut

    -- MIGRATED: Use newLayerIdx, removed stage check from selector
    -- Note: We're in a fold, need to get currentLayer from psSig for now
    selectedLd = mux ((processingLayer <$> psSig) .==. pure layerIndex)
                     ldNext
                     ldIn
  in
    (selectedLd, (writeDone, attnDone, qkvDone, qkvInReady, ffnDone), validOut, readyIn)
