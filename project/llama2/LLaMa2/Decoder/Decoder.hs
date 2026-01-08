module LLaMa2.Decoder.Decoder (
    decoder, DecoderIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), Temperature, Seed, Token)
import qualified Simulation.Parameters as PARAM (DecoderParameters (..), MultiHeadAttentionComponentQ (..), QueryHeadComponentQ (qMatrix))
import LLaMa2.Types.ModelConfig (NumLayers, ModelDimension, HeadDimension, SequenceLength)
import LLaMa2.Numeric.Types (FixedPoint, Mantissa)

import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (logitsProjector)
import qualified LLaMa2.Decoder.DataFlowController as Controller
import qualified LLaMa2.Decoder.LayerStack as LayerStack
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding
import qualified LLaMa2.Sampling.Sampler as Sampler

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import Simulation.Parameters (DecoderParameters(..), TransformerLayerComponent (multiHeadAttention))
import LLaMa2.Numeric.Operations (MultiplierState)
import LLaMa2.Numeric.Quantization (RowI8E(..))

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
  , paramQ0Row0         :: Signal dom Mantissa

  -- Debug fields propagated from LayerOutputs
  , dbgRowIndex         :: Signal dom (Index HeadDimension)
  , dbgState            :: Signal dom MultiplierState
  , dbgFirstMant        :: Signal dom Mantissa
  , dbgRowResult        :: Signal dom FixedPoint
  , dbgRowDone          :: Signal dom Bool
  , dbgFetchValid       :: Signal dom Bool
  , dbgFetchedWord       :: Signal dom (BitVector 512)
  , seqPos       :: Signal dom (Index SequenceLength)
  , cycleCount :: Signal dom (Unsigned 32)
  } deriving (Generic, NFDataX)

-- | Layer AXI arbiter: select active layer's AXI request
layerAxiArbiter :: forall dom.
  (KnownNat NumLayers)
  => Signal dom (Index NumLayers)
  -> Vec NumLayers (Master.AxiMasterOut dom)
  -> Master.AxiMasterOut dom
layerAxiArbiter activeLayerIdx axiMasters = Master.AxiMasterOut
  { arvalid = sel Master.arvalid
  , ardata  = sel Master.ardata
  , rready  = sel Master.rready
  , awvalid = sel Master.awvalid
  , awdata  = sel Master.awdata
  , wvalid  = sel Master.wvalid
  , wdata   = sel Master.wdata
  , bready  = sel Master.bready
  }
 where
  -- Helper that gathers a field from all masters into a Signal (Vec n a),
  -- then selects the element at runtime using the Signal index.
  sel :: forall a. (Master.AxiMasterOut dom -> Signal dom a) -> Signal dom a
  sel field =
    let fieldVec    :: Vec NumLayers (Signal dom a)
        fieldVec    = map field axiMasters

        fieldVecSig :: Signal dom (Vec NumLayers a)
        fieldVecSig = sequenceA fieldVec
    in (!!) <$> fieldVecSig <*> activeLayerIdx

-- | Main decoder with AXI interface
decoder :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)  -- Cycle counter
  -> Slave.AxiSlaveIn dom     -- DRAM interface
  -> PARAM.DecoderParameters
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Master.AxiMasterOut dom   -- AXI master out
     , Signal dom Token
     , Signal dom Bool
     , DecoderIntrospection dom )
decoder cycleCounter dramSlaveIn params inputToken forceInputToken temperature seed =
  (axiMasterOut, outputToken, readyPulse, introspection)
  where
    -- =======================================================================
    -- CONTROLLER
    -- =======================================================================
    controller = Controller.dataFlowController
      cycleCounter
      layerFfnDone     -- Layer processing complete
      logitsValid      -- Classifier/token complete

    processingStage = Controller.processingStage controller
    seqPosition     = Controller.seqPosition controller
    layerIdx        = Controller.currentLayer controller
    readyPulse      = Controller.readyPulse controller

    -- Extract enable signals from controller
    layerValid      = Controller.layerValidIn controller

    -- =======================================================================
    -- WEIGHT LOADING SYSTEM
    -- =======================================================================
    layerChanged = layerIdx ./=. register 0 layerIdx
    loadTrigger  = register True (pure False) .||. layerChanged

    -- =======================================================================
    -- LAYER PROCESSING (with AXI)
    -- =======================================================================
    embeddedVector :: Signal dom (Vec ModelDimension FixedPoint)
    embeddedVector = InputEmbedding.inputEmbedding (PARAM.modelEmbedding params) outputToken

    layerInput :: Signal dom LayerData
    layerInput = LayerStack.layerInputStage
      <$> layerIdx
      <*> layerDataReg  -- Use the explicitly named register
      <*> embeddedVector

    -- Capture the controller's layerValid signal from earlier
    controllerLayerValid = Controller.layerValidIn controller

    -- Latched version that holds until handshake completes
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

    -- Layer processing with AXI
    layerOutputs = LayerStack.activeLayerProcessor
      cycleCounter
      dramSlaveIn
      layerIdx
      seqPosition
      layerInput
      layerValidLatched
      params

    -- AXI arbitration
    axiMasterOut = layerAxiArbiter layerIdx (LayerStack.axiMasterOuts layerOutputs)

    nextLayerData :: Signal dom LayerData
    nextLayerData =
        mux layerFfnDone
          (LayerStack.ffnOutput layerOutputs)
          (mux layerAttnDone
              (LayerStack.attnOutput layerOutputs)
              (mux layerQkvDone
                  (LayerStack.qkvOutput layerOutputs)
                  layerDataReg))

    layerDataReg = register initialLayerData nextLayerData

    layerAttnDone = LayerStack.attnDone layerOutputs
    layerQkvDone  = LayerStack.qkvDone layerOutputs
    layerFfnDone  = LayerStack.ffnDone layerOutputs

    -- =======================================================================
    -- OUTPUT PROJECTION AND SAMPLING
    -- =======================================================================
    ffnOutput = feedForwardOutput <$> nextLayerData
    lastLayerComplete = (layerIdx .==. pure maxBound) .&&. layerFfnDone

    (logits, logitsValid) = OutputProjection.logitsProjector
      lastLayerComplete
      (pure True) -- always ready to consume (?)
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
      , loadTriggerActive   = loadTrigger
      , paramQ0Row0         = let
            mhaParams = multiHeadAttention $ head (modelLayers params)
            qMat = PARAM.qMatrix (head (PARAM.qHeads mhaParams))
            RowI8E {rowMantissas = mants, rowExponent =_exp} = qMat !! (0 :: Int)
          in pure $ head mants

      -- Propagate debug fields from layerOutputs
      , dbgRowIndex         = LayerStack.dbgRowIndex layerOutputs
      , dbgState            = LayerStack.dbgState layerOutputs
      , dbgFirstMant        = LayerStack.dbgFirstMant layerOutputs
      , dbgRowResult        = LayerStack.dbgRowResult layerOutputs
      , dbgRowDone          = LayerStack.dbgRowDone layerOutputs
      , dbgFetchValid       = LayerStack.dbgFetchValid layerOutputs
      , dbgFetchedWord       = LayerStack.dbgFetchedWord layerOutputs
      , seqPos = seqPosition
      , cycleCount = cycleCounter
      }
