-- | LLaMa2 Decoder - Top-level orchestration
module LLaMa2.Decoder.Decoder (
    decoder, DecoderIntrospection(..)
) where

import Clash.Prelude

import LLaMa2.Types.LayerData
  ( LayerData(..), ProcessingState (..), Temperature, Seed, Token )
import qualified Simulation.Parameters as PARAM (DecoderParameters (..))
import LLaMa2.Types.ModelConfig
  ( NumLayers, ModelDimension, NumKeyValueHeads, HeadDimension, HiddenDimension )
import LLaMa2.Numeric.Types (Mantissa)

import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (outputProjection)
import qualified LLaMa2.Decoder.SequenceController as SequenceController
import qualified LLaMa2.Decoder.LayerStack as LayerStack
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding
import qualified LLaMa2.Sampling.Sampler as Sampler

import LLaMa2.Memory.WeightLoader (weightManagementSystem, WeightSystemState)
import LLaMa2.Numeric.Quantization (RowI8E)

import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer (QKVProjectionWeightBuffer(..), qkvWeightBufferController)
import LLaMa2.Memory.LayerAddressing (LayerSeg(..), LayerAddress(..), layerAddressGenerator, WeightAddress (..), WeightMatrixType (..))
import LLaMa2.Memory.I8EDynamicRower (dynamicRower)
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn)
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut)

-- Initial state
initialLayerData :: LayerData
initialLayerData = LayerData
  { inputVector       = repeat 0
  , queryVectors      = repeat (repeat 0)
  , keyVectors        = repeat (repeat 0)
  , valueVectors      = repeat (repeat 0)
  , attentionOutput   = repeat 0
  , feedForwardOutput = repeat 0
  }

-- Introspection
data DecoderIntrospection dom = DecoderIntrospection
  { state               :: Signal dom ProcessingState
  , layerIndex          :: Signal dom (Index NumLayers)
  , ready               :: Signal dom Bool
  , attnDone            :: Signal dom Bool
  , ffnDone             :: Signal dom Bool
  , weightStreamValid   :: Signal dom Bool
  , parsedWeightSample  :: Signal dom Mantissa
  , layerChangeDetected :: Signal dom Bool
  , sysState            :: Signal dom WeightSystemState
  , weightBufferState   :: Signal dom QKVProjectionWeightBuffer
  } deriving (Generic, NFDataX)

decoder :: forall dom. HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Signal dom Bool
  -> PARAM.DecoderParameters
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , Master.AxiMasterOut dom
     , DecoderIntrospection dom )
decoder ddrSlave powerOn params inputToken inputTokenValid temperature seed =
  (outputToken, readyPulse, ddrMaster, introspection)
  where
    -- WEIGHT MANAGEMENT SYSTEM
    firstCycle = register True (pure False)
    prevLayer  = register 0 layerIdx
    layerChanged = layerIdx ./=. prevLayer
    loadTrigger = firstCycle .||. layerChanged

    startStream = powerOn .&&. loadTrigger

    -- Real weight manager backpressured by sinkReady (defined below)
    (  ddrMaster
     , weightStream
     , streamValid
     , sysState
     ) = weightManagementSystem ddrSlave startStream layerIdx sinkReady

    -- One real address generator, advanced by rowDoneExt from the rower
    (layerAddrSig, layerDone) = layerAddressGenerator rowDoneExt loadTrigger
    segSig = seg <$> layerAddrSig

    -- Dynamic rower (segment-aware, backpressured)
    (mdRowOut, mdRowValid, rowDoneExt, sinkReady) =
      dynamicRower (SNat @ModelDimension) (SNat @HeadDimension) (SNat @HiddenDimension)
                   weightStream streamValid segSig

    -- Hold last good MD row (safe sampling)
    parsedWeightsHold :: Signal dom (RowI8E ModelDimension)
    parsedWeightsHold = regEn (repeat 0, 0) mdRowValid mdRowOut

    -- Q/K/V buffering (map extended → legacy address; ignore when not Q/K/V)
    weightBuffer :: Signal dom QKVProjectionWeightBuffer
    weightBuffer =
      qkvWeightBufferController
        mdRowValid
        (mapToOldWeightAddr <$> layerAddrSig)
        parsedWeightsHold
        (qkvDonePulse <$> layerAddrSig <*> mdRowValid)
        loadTrigger

    mapToOldWeightAddr :: LayerAddress -> WeightAddress
    mapToOldWeightAddr la =
      case seg la of
        SegQ -> WeightAddress { rowIndex = rowIx la, matrixType = QMatrix, headIndex = resize (headIx la) }
        SegK -> WeightAddress { rowIndex = rowIx la, matrixType = KMatrix, headIndex = resize (headIx la) }
        SegV -> WeightAddress { rowIndex = rowIx la, matrixType = VMatrix, headIndex = resize (headIx la) }
        _    -> WeightAddress { rowIndex = 0, matrixType = QMatrix, headIndex = 0 }

    qkvDonePulse :: LayerAddress -> Bool -> Bool
    qkvDonePulse la vld =
      vld && seg la == SegV
          && headIx la == fromInteger (natToNum @NumKeyValueHeads - 1)
          && rowIx la  == maxBound

    -- Track load completion with proper reset
    loadComplete :: Signal dom Bool
    loadComplete = register False nextLoadComplete
      where
        nextLoadComplete = 
          mux loadTrigger
            (pure False)  -- Clear immediately on new load
            (mux (qkvDonePulse <$> layerAddrSig <*> mdRowValid)
              (pure True)   -- Set when loading completes
              loadComplete)  -- Maintain state

    enableAttention = loadComplete .&&. (fullyLoaded <$> weightBuffer)
    -- TODO replace useRAM with enableAttention
    useRAM :: Signal dom Bool
    useRAM = mux loadTrigger (pure False) ((fullyLoaded <$> weightBuffer) .&&. layerDone)
    
    (seqState, readyPulse) = SequenceController.sequenceController ffnDoneThisLayer
    layerIdx :: Signal dom (Index NumLayers)
    layerIdx = SequenceController.currentLayer <$> seqState

    processingState :: Signal dom ProcessingState
    processingState = SequenceController.processingState
      (SequenceController.pipelineController
        attnDoneThisLayer writeDoneThisLayer qkvDoneThisLayer
        ffnDoneThisLayer logitsValid inputTokenValid)

    outputToken = mux inputTokenValid inputToken feedbackToken
    embeddedVector = InputEmbedding.inputEmbedding (PARAM.modelEmbedding params) outputToken

    layerInput = LayerStack.prepareLayerInput <$> layerIdx
                   <*> register initialLayerData nextLayerData
                   <*> embeddedVector

    (nextLayerData, doneFlags) =
      LayerStack.processLayers processingState layerIdx layerInput weightBuffer useRAM (PARAM.modelLayers params)

    (writeDone, attnDone, qkvDone, _, ffnDone) = unzip5 doneFlags
    writeDoneThisLayer = LayerStack.getCurrentLayerFlag layerIdx writeDone
    attnDoneThisLayer  = LayerStack.getCurrentLayerFlag layerIdx attnDone
    qkvDoneThisLayer   = LayerStack.getCurrentLayerFlag layerIdx qkvDone
    ffnDoneThisLayer   = LayerStack.getCurrentLayerFlag layerIdx ffnDone

    layerOutput = feedForwardOutput <$> nextLayerData
    lastLayerComplete = (layerIdx .==. pure maxBound) .&&. ffnDoneThisLayer
    (logits, logitsValid) = OutputProjection.outputProjection params lastLayerComplete layerOutput
    sampledToken = Sampler.tokenSampler logitsValid temperature seed logits
    feedbackToken = regEn 0 logitsValid sampledToken

    -- monitoring
    (mantissasH, _expH) = unbundle parsedWeightsHold
    firstMantissa = fmap (bitCoerce . head) mantissasH
    introspection = DecoderIntrospection
      { state               = processingState
      , layerIndex          = layerIdx
      , ready               = readyPulse
      , attnDone            = attnDoneThisLayer
      , ffnDone             = ffnDoneThisLayer
      , weightStreamValid   = streamValid
      , parsedWeightSample  = firstMantissa
      , layerChangeDetected = layerChanged
      , sysState            = sysState
      , weightBufferState   = weightBuffer
      }
