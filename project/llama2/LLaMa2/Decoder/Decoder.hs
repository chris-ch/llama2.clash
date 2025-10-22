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
import LLaMa2.Numeric.Types (FixedPoint, Mantissa)

-- Import sub-modules
import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (outputProjection)
import qualified LLaMa2.Decoder.SequenceController as SequenceController
  ( pipelineController, processingState, sequenceController, SequenceState (..) )
import qualified LLaMa2.Decoder.LayerStack as LayerStack (processLayers, getCurrentLayerFlag, prepareLayerInput)
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding
import qualified LLaMa2.Sampling.Sampler as Sampler
import LLaMa2.Memory.AXI (AxiSlaveIn, AxiMasterOut)
import LLaMa2.Memory.WeightLoader (weightManagementSystem, parseI8EChunk, WeightSystemState, BootLoaderState)
import LLaMa2.Numeric.Quantization (RowI8E)

-- Step 1 addressing
import LLaMa2.Memory.WeightLoaderAddressing
  ( WeightAddress(..)
  , weightAddressGenerator
  )

-- Step 2: RAM weight buffer
import LLaMa2.Layer.Attention.WeightBuffer
  ( QKVWeightBuffer(..)
  , qkvWeightBufferController
  )

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
  { state               :: Signal dom ProcessingState
  , layerIndex          :: Signal dom (Index NumLayers)
  , ready               :: Signal dom Bool
  , attnDone            :: Signal dom Bool
  , ffnDone             :: Signal dom Bool
  , weightStreamValid   :: Signal dom Bool
  , parsedWeightSample  :: Signal dom Mantissa
  , bootProgressBytes   :: Signal dom (Unsigned 32)
  , layerChangeDetected :: Signal dom Bool
  , sysState            :: Signal dom WeightSystemState
  , bootState           :: Signal dom BootLoaderState
  -- Step 1
  , weightAddrDebug     :: Signal dom WeightAddress
  , qkvLoadDonePulse    :: Signal dom Bool
  -- Step 2
  , weightBufferState   :: Signal dom QKVWeightBuffer
  , useRAMFlag          :: Signal dom Bool
  } deriving (Generic, NFDataX)

-- ============================================================================
-- Main Decoder
-- ============================================================================

decoder :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool                     -- ^ bypass weight loading (for simulation)
  -> AxiSlaveIn dom                      -- ^ eMMC interface
  -> AxiSlaveIn dom                      -- ^ DDR4 interface  
  -> Signal dom Bool                     -- ^ Power on / boot trigger
  -> DecoderParameters
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , AxiMasterOut dom
     , AxiMasterOut dom
     , Signal dom Bool
     , Signal dom (Unsigned 32)
     , DecoderIntrospection dom
     )
decoder bypass emmcSlave ddrSlave powerOn params inputToken inputTokenValid temperature seed =
  (outputToken, readyPulse, emmcMaster, ddrMaster, weightsReady, bootProgress, introspection)
  where
    -- ========================================================================
    -- WEIGHT MANAGEMENT SYSTEM
    -- ========================================================================

    -- Detect layer changes to trigger weight streaming
    prevLayer = register 0 layerIdx
    layerChanged = layerIdx ./=. prevLayer

    -- Use test trigger instead of layerChanged for load
    loadTrigger = layerChanged .&&. weightsReady

    -- Instantiate the weight management system
    (emmcMaster, ddrMaster, weightStream, streamValid, weightsReady, bootProgress, sysState, bootState) =
      weightManagementSystem bypass emmcSlave ddrSlave powerOn layerIdx loadTrigger

    parsedWeights :: Signal dom (RowI8E ModelDimension)
    parsedWeights = parseI8EChunk <$> weightStream

    -- Step 1: addressing for each incoming row
    (weightAddrSig, qkvLoadDoneSig) =
      weightAddressGenerator streamValid loadTrigger

    -- Step 2: instantiate the weight buffer
    weightBuffer :: Signal dom QKVWeightBuffer
    weightBuffer =
      qkvWeightBufferController
        streamValid
        weightAddrSig
        parsedWeights
        qkvLoadDoneSig
        loadTrigger  -- reset the buffer when a new layer load starts

    useRAMWeights :: Signal dom Bool
    useRAMWeights = fullyLoaded <$> weightBuffer

    -- Extract first mantissa for debugging
    (mantissas, _exponent) = unbundle parsedWeights

    firstMantissa :: Signal dom Mantissa
    firstMantissa = fmap (bitCoerce . head) mantissas

    -- Gate input token until weights are ready (important!)
    gatedTokenValid = inputTokenValid .&&. weightsReady

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
        ffnDoneThisLayer logitsValid gatedTokenValid)

    -- ========================================================================
    -- TOKEN SELECTION & EMBEDDING
    -- ========================================================================

    outputToken :: Signal dom Token
    outputToken = mux gatedTokenValid inputToken feedbackToken

    embeddedVector :: Signal dom (Vec ModelDimension FixedPoint)
    embeddedVector = InputEmbedding.inputEmbedding (modelEmbedding params) outputToken

    -- ========================================================================
    -- LAYER STACK PROCESSING (unchanged for Step 2)
    -- ========================================================================

    layerInput :: Signal dom LayerData
    layerInput = LayerStack.prepareLayerInput <$> layerIdx
                   <*> register initialLayerData nextLayerData
                   <*> embeddedVector

    (nextLayerData, doneFlags) =
      LayerStack.processLayers
        processingState
        layerIdx
        layerInput
        parsedWeights
        streamValid
        (modelLayers params)

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
      { state               = processingState
      , layerIndex          = layerIdx
      , ready               = readyPulse
      , attnDone            = attnDoneThisLayer
      , ffnDone             = ffnDoneThisLayer
      , weightStreamValid   = streamValid
      , parsedWeightSample  = firstMantissa
      , bootProgressBytes   = bootProgress
      , layerChangeDetected = layerChanged
      , sysState            = sysState
      , bootState           = bootState
      , weightAddrDebug     = weightAddrSig
      , qkvLoadDonePulse    = qkvLoadDoneSig
      , weightBufferState   = weightBuffer         -- Step 2
      , useRAMFlag          = useRAMWeights        -- Step 2
      }
