module LLaMa2.Decoder.Decoder (
    decoder, DecoderIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), Temperature, Seed, Token)
import qualified Simulation.Parameters as PARAM (DecoderParameters (..), SingleHeadComponentQ (..), MultiHeadAttentionComponentQ (..))
import LLaMa2.Types.ModelConfig
  ( NumLayers, ModelDimension, NumKeyValueHeads, HeadDimension, HiddenDimension )
import LLaMa2.Numeric.Types (Mantissa, FixedPoint)

import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (logitsProjector)
import qualified LLaMa2.Decoder.DataFlowController as Controller
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
import qualified LLaMa2.Layer.Attention.QKVProjectionWeightBuffer as WEIGHTS
import Simulation.Parameters (DecoderParameters(..), TransformerLayerComponent (multiHeadAttention))

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
  { stage               :: Signal dom Controller.DataStage
  , layerIndex          :: Signal dom (Index NumLayers)
  , ready               :: Signal dom Bool
  , layerValidIn        :: Signal dom Bool
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
  , bufferFullyLoaded   :: Signal dom Bool
  , loadTriggerActive   :: Signal dom Bool
  , firstQMantissa   :: Signal dom (Signed 8)
  , rawWeightStream   :: Signal dom (BitVector 512)
  , mdRowOutSample   :: Signal dom (Vec ModelDimension Mantissa)
  , parsedWeightsHeld   :: Signal dom (Vec ModelDimension Mantissa)
  , firstMantissaFromRow   :: Signal dom (Signed 8)
  , paramQ0Row0  :: Signal dom (Signed 8)
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
decoder ddrSlave powerOn params inputToken forceInputToken temperature seed =
  (outputToken, readyPulse, ddrMaster, introspection)
  where
    -- =======================================================================
    -- CONTROLLER
    -- =======================================================================
    controller = Controller.dataFlowController
      layerFfnDone     -- Layer processing complete
      logitsValid      -- Classifier/token complete

    processingStage = Controller.processingStage controller
    seqPosition     = Controller.seqPosition controller
    layerIdx        = Controller.currentLayer controller
    readyPulse      = Controller.readyPulse controller

    -- Extract enable signals from controller
    layerValid       = Controller.layerValidIn controller

    -- =======================================================================
    -- WEIGHT LOADING SYSTEM
    -- =======================================================================
    layerChanged = layerIdx ./=. register 0 layerIdx
    loadTrigger  = register True (pure False) .||. layerChanged

    (ddrMaster, weightStream, streamValid, sysState) =
      weightManagementSystem ddrSlave (powerOn .&&. loadTrigger) layerIdx sinkReady

    (layerAddress, layerDone) = layerAddressGenerator rowDoneExt loadTrigger

    (mdRowOut, mdRowValid, rowDoneExt, sinkReady) =
      dynamicRower (SNat @ModelDimension) (SNat @HeadDimension) (SNat @HiddenDimension)
                   weightStream streamValid (seg <$> layerAddress)

    mdRowValidDelayed = register False (register False mdRowValid)  -- 2-cycle delay

    parsedWeightsHold :: Signal dom (RowI8E ModelDimension)
    parsedWeightsHold = regEn (repeat 0, 0) mdRowValidDelayed mdRowOut

    -- Data is already delayed by regEn, so delay address to match
    layerAddressDelayed :: Signal dom LayerAddress
    layerAddressDelayed = register (LayerAddress SegRmsAtt 0 0) layerAddress

    weightsBuffer :: Signal dom QKVProjectionWeightBuffer
    weightsBuffer = qkvWeightBufferController
      mdRowValid
      (mapToOldWeightAddr <$> layerAddressDelayed)  -- Now synchronized!
      parsedWeightsHold
      (qkvDonePulse <$> layerAddressDelayed <*> mdRowValid)
      loadTrigger

    -- =======================================================================
    -- LAYER PROCESSING (direct active layer)
    -- =======================================================================

    embeddedVector :: Signal dom (Vec ModelDimension FixedPoint)
    embeddedVector = InputEmbedding.inputEmbedding (PARAM.modelEmbedding params) outputToken

    layerInput :: Signal dom LayerData
    layerInput = LayerStack.prepareLayerInput
      <$> layerIdx
      <*> layerDataReg  -- Use the explicitly named register
      <*> embeddedVector

    -- Capture the controller's layerValid signal from earlier
    controllerLayerValid = Controller.layerValidIn controller

    -- Create a latched version that holds until handshake completes
    layerValidLatched :: Signal dom Bool
    layerValidLatched = register False nextLayerValidLatched
      where
        weightsReady = fullyLoaded <$> weightsBuffer
        layerReady = LayerStack.qkvReady layerOutputs

        -- Set: controller wants to start
        setLatch = controllerLayerValid .&&. (not <$> layerValidLatched)

        -- Clear: handshake completes (weights ready AND layer accepts)
        clearLatch = layerValidLatched .&&. weightsReady .&&. layerReady

        nextLayerValidLatched = mux setLatch (pure True)
                              (mux clearLatch (pure False) layerValidLatched)

    -- Only assert to layer when BOTH latched signal and weights are ready
    actualLayerValid :: Signal dom Bool
    actualLayerValid = layerValidLatched .&&. (fullyLoaded <$> weightsBuffer)

    -- Use actualLayerValid everywhere instead of layerValid
    layerOutputs = LayerStack.processActiveLayer
      layerIdx
      seqPosition
      layerInput
      actualLayerValid  -- <-- Use latched & gated signal
      weightsBuffer
      (PARAM.modelLayers params)

    nextLayerData :: Signal dom LayerData
    nextLayerData =
      mux layerFfnDone
        (LayerStack.ffnOutput layerOutputs)
        (mux layerAttnDone
            (LayerStack.attnOutput layerOutputs)
            (mux layerQkvDone
                (LayerStack.qkvOutput layerOutputs)
                layerDataReg))  -- Hold previous value when nothing completes

    layerDataReg = register initialLayerData nextLayerData

    layerAttnDone  = LayerStack.attnDone layerOutputs
    layerQkvDone   = LayerStack.qkvDone layerOutputs
    layerFfnDone   = LayerStack.ffnDone layerOutputs

    -- =======================================================================
    -- OUTPUT PROJECTION AND SAMPLING
    -- =======================================================================
    ffnOutput = feedForwardOutput <$> nextLayerData
    lastLayerComplete = (layerIdx .==. pure maxBound) .&&. layerFfnDone

    (logits, logitsValid) = OutputProjection.logitsProjector
      lastLayerComplete
      (pure True) -- always teady to consume (?)
      params
      ffnOutput

    sampledToken = Sampler.tokenSampler logitsValid temperature seed logits
    feedbackToken = regEn 0 logitsValid sampledToken

    -- =======================================================================
    -- TOKEN EMBEDDING AND FEEDBACK
    -- =======================================================================
    outputToken = mux forceInputToken inputToken feedbackToken

    -- =======================================================================
    -- INTROSPECTION
    -- =======================================================================
    (mantissasH, _expH) = unbundle parsedWeightsHold
    firstMantissa = bitCoerce . head <$> mantissasH

    introspection = DecoderIntrospection
      { stage               = processingStage
      , layerIndex          = layerIdx
      , ready               = readyPulse
      , layerValidIn        = layerValid
      , attnDone            = layerAttnDone
      , qkvDone             = layerQkvDone
      , ffnDone             = layerFfnDone
      , weightStreamValid   = streamValid
      , parsedWeightSample  = firstMantissa
      , layerChangeDetected = layerChanged
      , sysState            = sysState
      , weightBufferState   = weightsBuffer
      , layerOutput         = ffnOutput
      , layerData           = nextLayerData
      , bufferFullyLoaded = fullyLoaded <$> weightsBuffer
      , loadTriggerActive = loadTrigger  -- Should pulse on layer change
      , firstQMantissa = extractFirstMantissa <$> weightsBuffer  -- Track weight values
      , rawWeightStream = weightStream               -- Raw bytes from DDR
      , mdRowOutSample = fst <$> mdRowOut           -- Parsed row (before regEn)
      , parsedWeightsHeld = fst <$> parsedWeightsHold -- After regEn

      -- Helper to extract first mantissa from raw row
      , firstMantissaFromRow = head . fst <$> parsedWeightsHold
      , paramQ0Row0 = let
            mhaParams = multiHeadAttention $ head (modelLayers params)
            qMat = PARAM.wqHeadQ (head (PARAM.headsQ mhaParams))
            (mants, _exp) = qMat !! (0 :: Int)
            in pure $ head mants  -- First mantissa from hardcoded params
      }

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================
extractFirstMantissa :: QKVProjectionWeightBuffer -> Signed 8
extractFirstMantissa buf =
  let qWeight0 = WEIGHTS.extractQWeight buf 0
      firstRow = qWeight0 !! (0 :: Int)
      (mantissas, _) = firstRow
  in head mantissas

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
