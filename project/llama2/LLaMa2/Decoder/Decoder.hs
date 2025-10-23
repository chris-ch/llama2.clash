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
  , NumQueryHeads, NumKeyValueHeads, HeadDimension, HiddenDimension
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
import LLaMa2.Memory.WeightLoader (weightManagementSystem, WeightSystemState, BootLoaderState)
import LLaMa2.Numeric.Quantization (RowI8E)

import LLaMa2.Layer.Attention.WeightBuffer
  ( QKVWeightBuffer(..), qkvWeightBufferController )
import LLaMa2.Memory.WeightLoaderAddressingExtended
  ( LayerSeg(..), LayerAddress(..), layerAddressGenerator )
import LLaMa2.Memory.WeightLoaderAddressing (WeightAddress (..), WeightMatrixType (..))
import LLaMa2.Memory.I8EDynamicRower (dynamicRower)

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
    -- WEIGHT MANAGEMENT SYSTEM (extended, robust)
    -- ========================================================================

    -- Policies
    firstCycle :: Signal dom Bool
    firstCycle = register True (pure False)
    prevLayer  = register 0 layerIdx
    layerChanged = layerIdx ./=. prevLayer
    loadTrigger :: Signal dom Bool
    loadTrigger = firstCycle .||. layerChanged

    -- 1) Extended address generator seeded (we’ll rewire with real rowDoneExt)
    rowDoneExt_seed = pure False
    (layerAddrSig_seed, _layerAllDone_seed) =
      layerAddressGenerator rowDoneExt_seed loadTrigger

    segSig_seed = seg <$> layerAddrSig_seed

    -- 2) Weight manager (stub sinkReady=True)
    sinkReady_stub = pure True
    ( _emmcMaster_stub
     , _ddrMaster_stub
     , weightStream_stub
     , beatValid_stub
     , _weightsReady_stub
     , _bootProgress_stub
     , _sysState_stub
     , _bootState_stub
     ) = weightManagementSystem bypass emmcSlave ddrSlave powerOn layerIdx loadTrigger sinkReady_stub

    -- 3) Dynamic rower (segment-aware, backpressured)
    mdSNat  = SNat @ModelDimension
    hdSNat  = SNat @HeadDimension
    hidSNat = SNat @HiddenDimension
    (mdRowOut, mdRowValid, rowDoneExt, sinkReady) =
      dynamicRower mdSNat hdSNat hidSNat weightStream_stub beatValid_stub segSig_seed

    -- 4) Recreate address generator with real rowDoneExt
    (layerAddrSigR, _layerAllDoneR) =
      layerAddressGenerator rowDoneExt loadTrigger

    -- 5) Re-instantiate weight manager with real sinkReady (backpressured path)
    ( emmcMasterR
     , ddrMasterR
     , weightStreamR
     , beatValidR
     , weightsReadyR
     , bootProgressR
     , sysStateR
     , bootStateR
     ) = weightManagementSystem bypass emmcSlave ddrSlave powerOn layerIdx loadTrigger sinkReady

    -- Bind the “real” path to the names returned to Top
    emmcMaster   = emmcMasterR
    ddrMaster    = ddrMasterR
    weightsReady = weightsReadyR
    bootProgress = bootProgressR
    sysState     = sysStateR
    bootState    = bootStateR

    -- 6) Hold last good MD row for safe sampling
    parsedWeightsHold :: Signal dom (RowI8E ModelDimension)
    parsedWeightsHold = regEn (repeat 0, 0) mdRowValid mdRowOut

    -- 7) Q/K/V buffering with MD rows only (map extended address to legacy Q/K/V)
    weightBuffer :: Signal dom QKVWeightBuffer
    weightBuffer =
      qkvWeightBufferController
        mdRowValid
        (mapToOldWeightAddr <$> layerAddrSigR)
        parsedWeightsHold
        (qkvDonePulse <$> layerAddrSigR <*> mdRowValid)
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

    -- 8) Use-RAM switch: latched once QKV has completed
    useRAMEnable :: Signal dom Bool
    useRAMEnable = pure False
    useRAMLatched =
      let next = mux loadTrigger (pure False)
                 (useRAMLatched .||. (qkvDonePulse <$> layerAddrSigR <*> mdRowValid))
      in register False next
    useRAMWeights :: Signal dom Bool
    useRAMWeights = useRAMEnable .&&. useRAMLatched

    -- 9) Token gating; safe debug sample
    (mantissasH, _exponentH) = unbundle parsedWeightsHold
    firstMantissa :: Signal dom Mantissa
    firstMantissa = fmap (bitCoerce . head) mantissasH

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
    -- LAYER STACK PROCESSING
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
        weightBuffer
        useRAMWeights
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
      , weightStreamValid   = beatValidR
      , parsedWeightSample  = firstMantissa
      , bootProgressBytes   = bootProgress
      , layerChangeDetected = layerChanged
      , sysState            = sysState
      , bootState           = bootState
      , weightBufferState   = weightBuffer
      , useRAMFlag          = useRAMWeights
      }
