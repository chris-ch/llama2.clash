module LLaMa2.Decoder.Decoder (
    topEntity, decoder, DecoderIntrospection(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (ActivationBramAddr, Temperature, Seed, Token)
import LLaMa2.Types.ModelConfig (NumLayers, NumKeyValueHeads, ModelDimension)
import Data.Maybe (isJust)

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
    layerOutputs = LayerStack.activeLayerProcessor
      cycleCounter
      dramSlaveIn
      kvDramSlaves
      layerIdx
      seqPosition
      initWrPort
      layerValidLatched

    layerFfnDone = LayerStack.layerDone layerOutputs

    -- =======================================================================
    -- AXI arbitration
    -- =======================================================================
    axiMasterOut = Master.axiMasterMux classifierActive logitsAxiMaster
                 $ Master.axiMasterMux embBusy embAxiMaster
                 (LayerStack.axiMasterOut layerOutputs)

    -- =======================================================================
    -- FFN OUTPUT ACCUMULATOR
    -- Captures the slot-3 stream emitted by TransformerLayer during the copy
    -- phase and assembles it into a Vec for the logits projector.
    -- The stream runs for ModelDimension+2 cycles; we index by a counter that
    -- advances on each Just element.
    -- =======================================================================
    ffnStream = LayerStack.ffnStreamOut layerOutputs

    ffnStreamValid = isJust <$> ffnStream
    ffnStreamData  = (\mx -> case mx of { Just x -> x; Nothing -> 0 }) <$> ffnStream

    -- Detect the rising edge of ffnStreamValid so each new layer's stream
    -- resets the accumulation index to 0 (the copy phase emits 2 extra
    -- elements per layer due to prevCopyActive latency, drifting the counter
    -- by +2 per layer if not corrected).
    ffnStreamStart = ffnStreamValid .&&. (not <$> register False ffnStreamValid)

    -- Reset register to 1 on stream start (element 0 is written at index 0 via
    -- ffnEffectiveIdx override; subsequent elements use the register directly).
    ffnAccumCounter = register (0 :: Index ModelDimension) $
      mux ffnStreamStart (pure 1) $
      mux ffnStreamValid (satSucc SatWrap <$> ffnAccumCounter)
      ffnAccumCounter

    -- Override index to 0 on stream start so element 0 always lands at slot 0.
    ffnEffectiveIdx = mux ffnStreamStart (pure 0) ffnAccumCounter

    ffnOutputVec = register (repeat 0 :: Vec ModelDimension FixedPoint) $
      mux ffnStreamValid
        ((\vec i d -> replace i d vec) <$> ffnOutputVec <*> ffnEffectiveIdx <*> ffnStreamData)
        ffnOutputVec

    lastLayerComplete = (layerIdx .==. pure maxBound) .&&. layerFfnDone

    -- =======================================================================
    -- OUTPUT PROJECTION AND SAMPLING
    -- =======================================================================
    classifierActive = Controller.processingStage controller .==. pure Controller.Classifier

    (logitsAxiMaster, logitIdx, logitValue, logitValid, logitsAllDone) =
      OutputProjection.logitsProjector
        cycleCounter
        dramSlaveIn
        lastLayerComplete
        (pure True)
        logitsAllDone
        ffnOutputVec

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
      , ffnOut0      = fmap (!! (0 :: Index ModelDimension)) ffnOutputVec
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
