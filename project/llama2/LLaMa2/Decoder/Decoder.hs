module LLaMa2.Decoder.Decoder (
    decoder, DecoderIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), Temperature, Seed, Token)
import qualified Simulation.Parameters as PARAM (DecoderParameters (..), SingleHeadComponentQ (..), MultiHeadAttentionComponentQ (..))
import LLaMa2.Types.ModelConfig
  ( NumLayers, ModelDimension )
import LLaMa2.Numeric.Types (FixedPoint)

import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (logitsProjector)
import qualified LLaMa2.Decoder.DataFlowController as Controller
import qualified LLaMa2.Decoder.LayerStack as LayerStack
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding
import qualified LLaMa2.Sampling.Sampler as Sampler

import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn)
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
  , layerChangeDetected :: Signal dom Bool
  , layerOutput         :: Signal dom (Vec ModelDimension FixedPoint)
  , layerData           :: Signal dom LayerData
  , loadTriggerActive   :: Signal dom Bool
  , paramQ0Row0  :: Signal dom (Signed 8)
  } deriving (Generic, NFDataX)

-- | Main decoder with simplified control flow
decoder :: forall dom. HiddenClockResetEnable dom
  => PARAM.DecoderParameters
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , DecoderIntrospection dom )
decoder params inputToken forceInputToken temperature seed =
  (outputToken, readyPulse, introspection)
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
        layerReady = LayerStack.qkvReady layerOutputs

        -- Set: controller wants to start
        setLatch = controllerLayerValid .&&. (not <$> layerValidLatched)

        -- Clear: handshake completes (weights ready AND layer accepts)
        clearLatch = layerValidLatched .&&. layerReady

        nextLayerValidLatched = mux setLatch (pure True)
                              (mux clearLatch (pure False) layerValidLatched)

    -- Only assert to layer when BOTH latched signal and weights are ready
    actualLayerValid :: Signal dom Bool
    actualLayerValid = layerValidLatched

    -- Use actualLayerValid everywhere instead of layerValid
    layerOutputs = LayerStack.processActiveLayer
      layerIdx
      seqPosition
      layerInput
      actualLayerValid  -- <-- Use latched & gated signal
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

    introspection = DecoderIntrospection
      { stage               = processingStage
      , layerIndex          = layerIdx
      , ready               = readyPulse
      , layerValidIn        = layerValid
      , attnDone            = layerAttnDone
      , qkvDone             = layerQkvDone
      , ffnDone             = layerFfnDone
      , layerChangeDetected = layerChanged
      , layerOutput         = ffnOutput
      , layerData           = nextLayerData
      , loadTriggerActive = loadTrigger  -- Should pulse on layer change

      , paramQ0Row0 = let
            mhaParams = multiHeadAttention $ head (modelLayers params)
            qMat = PARAM.wqHeadQ (head (PARAM.headsQ mhaParams))
            (mants, _exp) = qMat !! (0 :: Int)
            in pure $ head mants  -- First mantissa from hardcoded params
      }
