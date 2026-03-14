module LLaMa2.Decoder.Decoder (
    topEntity, decoder, DecoderIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..), Temperature, Seed, Token)
import LLaMa2.Types.ModelConfig (NumLayers, NumKeyValueHeads)

import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (logitsProjector)
import qualified LLaMa2.Decoder.DataFlowController as Controller
import qualified LLaMa2.Decoder.LayerRunner as LayerStack
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding
import qualified LLaMa2.Sampling.Sampler as Sampler

import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn(..))
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut(..), axiMasterMux)

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

-- | Simulation introspection signals
data DecoderIntrospection dom = DecoderIntrospection
  { layerIndex  :: Signal dom (Index NumLayers)
  , ready       :: Signal dom Bool
  , layerValidIn :: Signal dom Bool
  , attnDone    :: Signal dom Bool
  , qkvDone     :: Signal dom Bool
  , ffnDone     :: Signal dom Bool
  , cycleCount  :: Signal dom (Unsigned 32)
  } deriving (Generic, NFDataX)

-- | Main decoder with AXI interface
decoder :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                                         -- Cycle counter
  -> Slave.AxiSlaveIn dom                                             -- weights DRAM
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)                      -- KV cache DRAM
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Master.AxiMasterOut dom                                         -- weights AXI master
     , Vec NumKeyValueHeads (Master.AxiMasterOut dom)                 -- KV cache AXI masters
     , Signal dom Token
     , Signal dom Bool
     , DecoderIntrospection dom )
decoder cycleCounter dramSlaveIn kvDramSlaves inputToken forceInputToken temperature seed =
  (axiMasterOut, LayerStack.kvAxiMasterOuts layerOutputs, outputToken, readyPulse, introspection)
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
    layerValid      = Controller.layerValidIn controller

    -- =======================================================================
    -- LAYER PROCESSING (with AXI)
    -- =======================================================================

    -- Trigger embedding fetch one cycle after token is stable
    firstCycle :: Signal dom Bool
    firstCycle = register True (pure False)

    embFetchTrigger :: Signal dom Bool
    embFetchTrigger = firstCycle .||. register False readyPulse

    (embAxiMaster, embeddedVector, embOutputValid, embBusy) =
      InputEmbedding.inputEmbedding
        cycleCounter dramSlaveIn
        embFetchTrigger outputToken

    layerInput :: Signal dom LayerData
    layerInput = LayerStack.layerInputStage
      <$> layerIdx
      <*> layerDataReg
      <*> embeddedVector

    -- Capture the controller's layerValid signal from earlier
    controllerLayerValid = Controller.layerValidIn controller

    -- Embedding is ready for layer 0: valid and not mid-fetch
    embeddingOk :: Signal dom Bool
    embeddingOk = embOutputValid .&&. (not <$> embBusy)

    -- Hold layer-0 start request until embedding is ready.
    -- Other layers start immediately on controllerLayerValid.
    pendingLayer0 :: Signal dom Bool
    pendingLayer0 = register False nextPendingLayer0
      where
        consumed = pendingLayer0 .&&. embeddingOk .&&. (not <$> layerValidLatched)
        nextPendingLayer0 =
          mux consumed (pure False) $
          mux (controllerLayerValid .&&. (layerIdx .==. 0)) (pure True)
          pendingLayer0

    -- Latched version that holds until handshake completes
    layerValidLatched :: Signal dom Bool
    layerValidLatched = register False nextLayerValidLatched
      where
        layerReady = LayerStack.qkvReady layerOutputs

        -- Layer 0: wait for embedding; other layers: fire immediately
        effectiveValid =
          ((layerIdx ./=. 0) .&&. controllerLayerValid) .||.
          (pendingLayer0 .&&. embeddingOk)

        setLatch  = effectiveValid .&&. (not <$> layerValidLatched)
        clearLatch = layerValidLatched .&&. layerReady

        nextLayerValidLatched = mux setLatch (pure True)
                              (mux clearLatch (pure False) layerValidLatched)

    -- Layer processing with AXI
    layerOutputs = LayerStack.activeLayerProcessor
      cycleCounter
      dramSlaveIn
      kvDramSlaves
      layerIdx
      seqPosition
      layerInput
      layerValidLatched

    -- AXI arbitration: embedding has priority while fetching
    axiMasterOut = Master.axiMasterMux classifierActive logitsAxiMaster
                 $ Master.axiMasterMux embBusy embAxiMaster (LayerStack.axiMasterOut layerOutputs)

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

    classifierActive = processingStage .==. pure Controller.Classifier

    (logitsAxiMaster, logits, logitsValid) = OutputProjection.logitsProjector
      cycleCounter
      dramSlaveIn
      lastLayerComplete
      (pure True) -- always ready to consume
      logitsValid -- self-consuming: OTC clears on next cycle after outputValid
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
      { layerIndex  = layerIdx
      , ready       = readyPulse
      , layerValidIn = layerValid
      , attnDone    = layerAttnDone
      , qkvDone     = layerQkvDone
      , ffnDone     = layerFfnDone
      , cycleCount  = cycleCounter
      }

-- | Synthesis wrapper: generates cycle counter internally, drops introspection
decoderTop :: HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Master.AxiMasterOut dom
     , Vec NumKeyValueHeads (Master.AxiMasterOut dom)
     , Signal dom Token
     , Signal dom Bool
     )
decoderTop dramSlaveIn kvDramSlaves inputToken forceInputToken temperature seed =
  (axiOut, kvOut, tokenOut, readyOut)
  where
    cycleCounter   = register (0 :: Unsigned 32) (cycleCounter + 1)
    (axiOut, kvOut, tokenOut, readyOut, _) =
      decoder cycleCounter dramSlaveIn kvDramSlaves inputToken forceInputToken temperature seed

{-# ANN topEntity
  (Synthesize
    { t_name   = "decoder"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortProduct "weights_dram" []
        , PortProduct "kv_dram"      []
        , PortName "input_token"
        , PortName "force_input_token"
        , PortName "temperature"
        , PortName "seed"
        ]
    , t_output =
        PortProduct ""
          [ PortProduct "weights_axi" []
          , PortProduct "kv_axi"      []
          , PortName "output_token"
          , PortName "ready"
          ]
    }) #-}
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Slave.AxiSlaveIn System
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn System)
  -> Signal System Token
  -> Signal System Bool
  -> Signal System Temperature
  -> Signal System Seed
  -> ( Master.AxiMasterOut System
     , Vec NumKeyValueHeads (Master.AxiMasterOut System)
     , Signal System Token
     , Signal System Bool
     )
topEntity = exposeClockResetEnable decoderTop
