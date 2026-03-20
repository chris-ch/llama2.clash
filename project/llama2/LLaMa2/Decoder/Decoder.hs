module LLaMa2.Decoder.Decoder (
    topEntity, decoder, DecoderIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (ActivationBramAddr, Temperature, Seed, Token)
import LLaMa2.Types.ModelConfig (NumLayers, NumKeyValueHeads, ModelDimension)


import qualified LLaMa2.Embedding.OutputProjection as OutputProjection (logitsProjector)
import qualified LLaMa2.Sampling.Sampler as Sampler (tokenSampler)
import qualified LLaMa2.Decoder.DataFlowController as Controller
import qualified LLaMa2.Decoder.LayerRunner as LayerStack
import qualified LLaMa2.Embedding.InputEmbedding as InputEmbedding

import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn(..))
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut(..), axiMasterMux)
import LLaMa2.Numeric.Types (FixedPoint)

-- | Simulation introspection signals
data DecoderIntrospection dom = DecoderIntrospection
  { layerIndex     :: Signal dom (Index NumLayers)
  , ready          :: Signal dom Bool
  , layerValidIn   :: Signal dom Bool
  , layerDone      :: Signal dom Bool
  , cycleCount     :: Signal dom (Unsigned 32)
  , ffnOut0        :: Signal dom FixedPoint  -- ^ ffnOutputVec[0] (debug: check residual correctness)
  } deriving (Generic, NFDataX)

-- | Main decoder with AXI interface
decoder :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                                         -- Cycle counter
  -> Slave.AxiSlaveIn dom                                             -- weights DRAM
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)                      -- KV cache DRAM
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Bool                                                   -- ^ softReset
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Master.AxiMasterOut dom
     , Vec NumKeyValueHeads (Master.AxiMasterOut dom)
     , Signal dom Token
     , Signal dom Bool
     , DecoderIntrospection dom )
decoder cycleCounter dramSlaveIn kvDramSlaves inputToken forceInputToken softReset temperature seed =
  (axiMasterOut, LayerStack.kvAxiMasterOuts layerOutputs, outputToken, readyPulse, introspection)
  where
    -- =======================================================================
    -- CONTROLLER
    -- =======================================================================
    controller = Controller.dataFlowController
      softReset
      layerFfnDone
      samplerValid

    seqPosition = Controller.seqPosition controller
    layerIdx    = Controller.currentLayer controller
    readyPulse  = Controller.readyPulse controller

    controllerLayerValid = Controller.layerValidIn controller

    -- =======================================================================
    -- INPUT EMBEDDING FETCH
    -- =======================================================================
    firstCycle :: Signal dom Bool
    firstCycle = register True (pure False)

    embFetchTrigger :: Signal dom Bool
    embFetchTrigger = firstCycle .||. register False readyPulse

    (embAxiMaster, embeddedVector, embOutputValid, embBusy) =
      InputEmbedding.inputEmbedding
        cycleCounter dramSlaveIn
        embFetchTrigger outputToken

    -- =======================================================================
    -- SLOT 0 INIT: sequential write of embedding to BRAM slot 0 (layer 0 only)
    -- The TransformerLayer auto-copies slot 3 → slot 0 between layers,
    -- so initWrPort is only needed for layer 0's embedding.
    -- =======================================================================
    -- Trigger: layer 0 is starting AND embedding is ready AND layer is idle.
    layer0Start :: Signal dom Bool
    layer0Start = (layerIdx .==. 0) .&&. embeddingOk .&&. LayerStack.readyOut layerOutputs

    embeddingOk = embOutputValid .&&. (not <$> embBusy)

    -- Sequential counter that runs for ModelDimension cycles to write embedding.
    initActive = register False $
      mux initAtMax (pure False) $
      mux layer0StartPulse (pure True)
      initActive

    layer0StartPulse = layer0Start .&&. (not <$> register False layer0Start)

    initCounter = register (0 :: Index ModelDimension) $
      mux initActive (satSucc SatWrap <$> initCounter) (pure 0)

    initAtMax = initActive .&&. initCounter .==. pure maxBound

    initDone = register False initAtMax

    -- Drive initWrPort: write embedding[i] to slot 0 addr i.
    initWrPort :: Signal dom (Maybe (ActivationBramAddr, FixedPoint))
    initWrPort = mux initActive
      (Just <$> ((,) <$> (fromIntegral <$> initCounter)
                       <*> ((!!) <$> embeddedVector <*> initCounter)))
      (pure Nothing)

    -- =======================================================================
    -- LAYER VALID: fire after init is done (layer 0) or immediately (layer N>0)
    -- =======================================================================
    pendingLayer0 = register False $
      mux (pendingLayer0 .&&. initDone) (pure False) $
      mux (controllerLayerValid .&&. (layerIdx .==. 0)) (pure True)
      pendingLayer0

    layerValidLatched = register False nextLayerValidLatched
      where
        layerReady = LayerStack.readyOut layerOutputs
        effectiveValid =
          ((layerIdx ./=. 0) .&&. controllerLayerValid) .||.
          (pendingLayer0 .&&. initDone)
        setLatch   = effectiveValid .&&. (not <$> layerValidLatched)
        clearLatch = layerValidLatched .&&. layerReady
        nextLayerValidLatched = mux setLatch (pure True)
                              (mux clearLatch (pure False) layerValidLatched)

    -- =======================================================================
    -- LAYER PROCESSING
    -- =======================================================================
    -- =======================================================================
    -- LAYER PROCESSING
    -- logitsBramRdAddr (from logitsProjector) drives the activation BRAM
    -- read port when the layer is idle, allowing the logits projector to read
    -- slot 3 directly without accumulating a Vec register.
    -- =======================================================================
    layerOutputs = LayerStack.activeLayerProcessor
      cycleCounter
      dramSlaveIn
      kvDramSlaves
      layerIdx
      seqPosition
      initWrPort
      layerValidLatched
      logitsBramRdAddr

    layerFfnDone = LayerStack.layerDone layerOutputs

    lastLayerComplete = (layerIdx .==. pure maxBound) .&&. layerFfnDone

    -- =======================================================================
    -- AXI arbitration
    -- =======================================================================
    axiMasterOut = Master.axiMasterMux classifierActive logitsAxiMaster
                 $ Master.axiMasterMux embBusy embAxiMaster
                 (LayerStack.axiMasterOut layerOutputs)

    -- =======================================================================
    -- OUTPUT PROJECTION AND SAMPLING
    -- =======================================================================
    classifierActive = Controller.processingStage controller .==. pure Controller.Classifier

    (logitsAxiMaster, logitsBramRdAddr, logitIdx, logitValue, logitValid, logitsAllDone) =
      OutputProjection.logitsProjector
        cycleCounter
        dramSlaveIn
        lastLayerComplete
        (pure True)
        logitsAllDone
        (LayerStack.bramRdDataOut layerOutputs)

    (sampledToken, samplerValid) =
      Sampler.tokenSampler logitIdx logitValue logitValid logitsAllDone temperature seed

    feedbackToken = regEn 0 samplerValid sampledToken

    -- =======================================================================
    -- TOKEN EMBEDDING AND FEEDBACK
    -- =======================================================================
    outputToken = mux forceInputToken inputToken feedbackToken

    -- =======================================================================
    -- INTROSPECTION
    -- =======================================================================
    introspection = DecoderIntrospection
      { layerIndex   = layerIdx
      , ready        = readyPulse
      , layerValidIn = controllerLayerValid
      , layerDone    = layerFfnDone
      , cycleCount   = cycleCounter
      , ffnOut0      = LayerStack.ffnOut0 layerOutputs
      }

-- | Synthesis wrapper
decoderTop :: HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Master.AxiMasterOut dom
     , Vec NumKeyValueHeads (Master.AxiMasterOut dom)
     , Signal dom Token
     , Signal dom Bool
     )
decoderTop dramSlaveIn kvDramSlaves inputToken forceInputToken softReset temperature seed =
  (axiOut, kvOut, tokenOut, readyOut)
  where
    cycleCounter   = register (0 :: Unsigned 32) (cycleCounter + 1)
    (axiOut, kvOut, tokenOut, readyOut, _) =
      decoder cycleCounter dramSlaveIn kvDramSlaves inputToken forceInputToken softReset temperature seed

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
        , PortName "soft_reset"
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
  -> Signal System Bool
  -> Signal System Temperature
  -> Signal System Seed
  -> ( Master.AxiMasterOut System
     , Vec NumKeyValueHeads (Master.AxiMasterOut System)
     , Signal System Token
     , Signal System Bool
     )
topEntity = exposeClockResetEnable decoderTop
