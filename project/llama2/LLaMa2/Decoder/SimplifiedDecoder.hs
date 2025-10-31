module LLaMa2.Decoder.SimplifiedDecoder (
    decoder, DecoderIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), ProcessingState, Temperature, Seed, Token)
import qualified Simulation.Parameters as PARAM (DecoderParameters (..))
import LLaMa2.Types.ModelConfig
  ( NumLayers, ModelDimension, NumKeyValueHeads, HeadDimension, HiddenDimension )
import LLaMa2.Numeric.Types (Mantissa, FixedPoint)

import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (outputProjection)
import qualified LLaMa2.Decoder.SimplifiedSequenceController as Controller
import qualified LLaMa2.Decoder.SimplifiedLayerStack as LayerStack
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding
import qualified LLaMa2.Sampling.Sampler as Sampler

import LLaMa2.Memory.WeightLoader (weightManagementSystem, WeightSystemState)
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer (QKVProjectionWeightBuffer(..), qkvWeightBufferController)
import LLaMa2.Memory.LayerAddressing (LayerSeg(..), LayerAddress(..), layerAddressGenerator, WeightAddress (..), WeightMatrixType (..))
import LLaMa2.Memory.I8EDynamicRower (dynamicRower)
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn)
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut)

-- | Initial layer data (all zeros)
initialLayerData :: LayerData
initialLayerData = LayerData
  { inputVector       = repeat 0
  , queryVectors      = repeat (repeat 0)
  , keyVectors        = repeat (repeat 0)
  , valueVectors      = repeat (repeat 0)
  , attentionOutput   = repeat 0
  , feedForwardOutput = repeat 0
  }

-- | Introspection signals for debugging
data DecoderIntrospection dom = DecoderIntrospection
  { state               :: Signal dom ProcessingState
  , layerIndex          :: Signal dom (Index NumLayers)
  , ready               :: Signal dom Bool
  , attnDone            :: Signal dom Bool
  , qkvDone             :: Signal dom Bool
  , ffnDone             :: Signal dom Bool
  , weightStreamValid   :: Signal dom Bool
  , parsedWeightSample  :: Signal dom Mantissa
  , layerChangeDetected :: Signal dom Bool
  , sysState            :: Signal dom WeightSystemState
  , weightBufferState   :: Signal dom QKVProjectionWeightBuffer
  , layerOutput         :: Signal dom (Vec ModelDimension FixedPoint)
  , layerData           :: Signal dom LayerData
  } deriving (Generic, NFDataX)

-- | Main decoder with simplified control flow
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
    -- =======================================================================
    -- CONTROLLER
    -- =======================================================================
    controller = Controller.unifiedController
      layerQkvDone layerWriteDone layerAttnDone layerFfnDone logitsValid

    processingState = Controller.processingState controller
    seqPos          = Controller.seqPosition controller  -- Extract directly
    layerIdx        = Controller.currentLayer controller
    readyPulse      = Controller.readyPulse controller

    -- NEW: Extract enable signals from controller
    enableQKV       = Controller.enableQKV controller
    enableWriteKV   = Controller.enableWriteKV controller
    enableAttend    = Controller.enableAttend controller
    enableFFN       = Controller.enableFFN controller
    enableClassifier = Controller.enableClassifier controller
    
    -- =======================================================================
    -- WEIGHT LOADING SYSTEM
    -- =======================================================================
    layerChanged = layerIdx ./=. register 0 layerIdx
    loadTrigger  = register True (pure False) .||. layerChanged
    
    (ddrMaster, weightStream, streamValid, sysState) =
      weightManagementSystem ddrSlave (powerOn .&&. loadTrigger) layerIdx sinkReady

    (layerAddrSig, layerDone) = layerAddressGenerator rowDoneExt loadTrigger
    
    (mdRowOut, mdRowValid, rowDoneExt, sinkReady) =
      dynamicRower (SNat @ModelDimension) (SNat @HeadDimension) (SNat @HiddenDimension)
                   weightStream streamValid (seg <$> layerAddrSig)

    parsedWeightsHold :: Signal dom (RowI8E ModelDimension)
    parsedWeightsHold = regEn (repeat 0, 0) mdRowValid mdRowOut

    -- QKV weight buffer
    weightBuffer = qkvWeightBufferController
      mdRowValid
      (mapToOldWeightAddr <$> layerAddrSig)
      parsedWeightsHold
      (qkvDonePulse <$> layerAddrSig <*> mdRowValid)
      loadTrigger

    -- Weight loading complete when QKV buffer is fully loaded
    useRAM = (fullyLoaded <$> weightBuffer) .&&. layerDone

    -- =======================================================================
    -- TOKEN EMBEDDING AND FEEDBACK
    -- =======================================================================
    outputToken = mux inputTokenValid inputToken feedbackToken
    embeddedVector = InputEmbedding.inputEmbedding (PARAM.modelEmbedding params) outputToken

    -- =======================================================================
    -- LAYER PROCESSING (direct active layer)
    -- =======================================================================
    layerInput = LayerStack.prepareLayerInput 
      <$> layerIdx 
      <*> register initialLayerData nextLayerData 
      <*> embeddedVector

    -- Extract seqPos for modules that need it, but keep processingState too
    layerOutput = LayerStack.processActiveLayer
      processingState layerIdx layerInput weightBuffer useRAM (PARAM.modelLayers params)
      enableQKV enableWriteKV enableAttend enableFFN enableClassifier
    
    nextLayerData   = LayerStack.outputData layerOutput
    layerWriteDone  = LayerStack.writeDone layerOutput
    layerAttnDone   = LayerStack.attnDone layerOutput
    layerQkvDone    = LayerStack.qkvDone layerOutput
    layerFfnDone    = LayerStack.ffnDone layerOutput

    -- =======================================================================
    -- OUTPUT PROJECTION AND SAMPLING
    -- =======================================================================
    ffnOutput = feedForwardOutput <$> nextLayerData
    lastLayerComplete = (layerIdx .==. pure maxBound) .&&. layerFfnDone

    (logits, logitsValid) = OutputProjection.outputProjection params lastLayerComplete ffnOutput

    sampledToken = Sampler.tokenSampler logitsValid temperature seed logits
    feedbackToken = regEn 0 logitsValid sampledToken

    -- =======================================================================
    -- INTROSPECTION
    -- =======================================================================
    (mantissasH, _expH) = unbundle parsedWeightsHold
    firstMantissa = fmap (bitCoerce . head) mantissasH
    
    introspection = DecoderIntrospection
      { state               = processingState
      , layerIndex          = layerIdx
      , ready               = readyPulse
      , attnDone            = layerAttnDone
      , qkvDone             = layerQkvDone
      , ffnDone             = layerFfnDone
      , weightStreamValid   = streamValid
      , parsedWeightSample  = firstMantissa
      , layerChangeDetected = layerChanged
      , sysState            = sysState
      , weightBufferState   = weightBuffer
      , layerOutput         = ffnOutput
      , layerData           = nextLayerData
      }

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

mapToOldWeightAddr :: LayerAddress -> WeightAddress
mapToOldWeightAddr la = case seg la of
  SegQ -> WeightAddress { rowIndex = rowIx la, matrixType = QMatrix, headIndex = resize (headIx la) }
  SegK -> WeightAddress { rowIndex = rowIx la, matrixType = KMatrix, headIndex = resize (headIx la) }
  SegV -> WeightAddress { rowIndex = rowIx la, matrixType = VMatrix, headIndex = resize (headIx la) }
  _    -> WeightAddress { rowIndex = 0, matrixType = QMatrix, headIndex = 0 }

qkvDonePulse :: LayerAddress -> Bool -> Bool
qkvDonePulse la vld =
  vld && seg la == SegV
      && headIx la == fromInteger (natToNum @NumKeyValueHeads - 1)
      && rowIx la == maxBound
