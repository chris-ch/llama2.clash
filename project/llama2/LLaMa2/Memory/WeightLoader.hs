module LLaMa2.Memory.WeightLoader 
 (
  calculateLayerBaseAddress,
  calculateLayerSizeBytes,
  --layerWeightStreamer,
  parseI8EChunk,
  weightManagementSystem,
  WeightSystemState(..)
 )
    
    where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Numeric.Types (Exponent, Mantissa)
import LLaMa2.Memory.AxiReadMaster (axiBurstReadMaster)
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn)
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut (..))

-- ============================================================================
-- File Layout Constants (matches Parser.hs)
-- ============================================================================

-- Calculate DDR4 base address for a specific layer
calculateLayerBaseAddress :: Index NumLayers -> Unsigned 32
calculateLayerBaseAddress layerIdx = embeddingSize + rmsFinalSize + rotarySize + layerOffset
  where
    vocabSize = snatToNum (SNat @VocabularySize)
    modelDim = snatToNum (SNat @ModelDimension)
    seqLen = snatToNum (SNat @SequenceLength)
    rotaryDim = snatToNum (SNat @RotaryPositionalEmbeddingDimension)

    embeddingSize = vocabSize * (modelDim + 1)
    rmsFinalSize = modelDim + 1
    rotarySize = 2 * seqLen * rotaryDim * 4
    layerOffset = fromIntegral layerIdx * calculateLayerSizeBytes

-- Calculate size of one layer in bytes (I8E format: 1 byte mantissa + 1/N byte exponent per row)
calculateLayerSizeBytes :: Unsigned 32
calculateLayerSizeBytes = rmsAttSize + wqSize + wkSize + wvSize + woSize +
                          rmsFfnSize + w1Size + w2Size + w3Size
  where
    modelDim = snatToNum (SNat @ModelDimension)
    hiddenDim = snatToNum (SNat @HiddenDimension)
    numQueryHeads = snatToNum (SNat @NumQueryHeads)
    numKVHeads = snatToNum (SNat @NumKeyValueHeads)
    headDim = snatToNum (SNat @HeadDimension)

    -- Each row: N mantissas (1 byte each) + 1 exponent (1 byte)
    rmsAttSize = modelDim + 1
    wqSize = numQueryHeads * headDim * (modelDim + 1)
    wkSize = numKVHeads * headDim * (modelDim + 1)
    wvSize = numKVHeads * headDim * (modelDim + 1)
    woSize = modelDim * (modelDim + 1)
    rmsFfnSize = modelDim + 1
    w1Size = hiddenDim * (modelDim + 1)
    w2Size = modelDim * (hiddenDim + 1)
    w3Size = hiddenDim * (modelDim + 1)

-- ============================================================================
-- RUNTIME STREAMER: DDR4 → FPGA (Layer-at-a-time)
-- ============================================================================

data StreamerState
  = StreamIdle
  | StreamBursting   -- Streaming bursts
  | StreamComplete
  deriving (Generic, NFDataX, Show, Eq)

-- | Stream one layer's weights from DDR4 (READY-aware)
layerWeightStreamer
  :: forall dom . HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom                     -- DDR4 AXI slave
  -> Signal dom (Index NumLayers)       -- Layer to load
  -> Signal dom Bool                    -- Start load
  -> Signal dom Bool                    -- sinkReady (consumer can accept a beat)
  -> ( Master.AxiMasterOut dom                 -- DDR4 AXI master
     , Signal dom (BitVector 512)       -- Raw weight data (64 bytes)
     , Signal dom Bool                  -- Data valid
     , Signal dom Bool                  -- Load complete
     )
layerWeightStreamer ddrSlave layerIdx startLoad sinkReady =
  (axiMaster, weightData, dataValid, loadComplete)
  where
    state = register StreamIdle nextState

    layerBaseAddr = calculateLayerBaseAddress <$> layerIdx

    burstLen   = 255 :: Unsigned 8
    burstBytes = 16384 :: Unsigned 32
    totalBursts = (calculateLayerSizeBytes + burstBytes - 1) `div` burstBytes

    currentBurst = register (0 :: Unsigned 32) nextBurst
    burstAddr    = layerBaseAddr + (currentBurst * pure burstBytes)

    startBurst = state .==. pure StreamBursting
    (axiMaster, weightData, dataValid, ready) =
      axiBurstReadMaster ddrSlave burstAddr (pure burstLen) startBurst sinkReady

    transfersInBurst = register (0 :: Unsigned 8) nextTransferCount
    burstComplete = register False $
      mux (state ./=. pure StreamBursting) (pure False) $
      mux (transfersInBurst .==. pure 255) (pure True) burstComplete

    -- do not reset while in StreamBursting each cycle; reset only on Idle or when a burst completes
    nextTransferCount =
      mux (state .==. pure StreamIdle) 0 $
      mux burstComplete 0 $
      mux dataValid (transfersInBurst + 1) transfersInBurst

    allBurstsComplete = currentBurst .>=. pure totalBursts

    nextState =
      mux (state .==. pure StreamIdle)
        (mux startLoad (pure StreamBursting) (pure StreamIdle)) $
      mux (state .==. pure StreamBursting)
        (mux (burstComplete .&&. allBurstsComplete) (pure StreamComplete)
          (mux burstComplete (pure StreamBursting) (pure StreamBursting))) $
      -- StreamComplete -> StreamIdle
      pure StreamIdle

    nextBurst =
      mux (state .==. pure StreamIdle) 0 $
      mux burstComplete (currentBurst + 1) currentBurst

    loadComplete = state .==. pure StreamComplete

-- ============================================================================
-- PARSER: BitVector 512 → I8E Format (No dequantization!)
-- ============================================================================

-- | Parse 512-bit chunk into I8E row format
-- 512 bits = 64 bytes = 63 mantissas + 1 exponent (for rows ≤ 63 elements)
parseI8EChunk
  :: forall n . (KnownNat n)
  => BitVector 512
  -> RowI8E n
parseI8EChunk rawData = (mantissas, expon)
  where
    -- Convert BitVector 512 to Vec 64 (BitVector 8)
    bytes = bitCoerce rawData :: Vec 64 (BitVector 8)

    -- Extract first N mantissas using map over indices
    mantissas = map (\i -> bitCoerce (bytes !! i)) indicesI :: Vec n Mantissa

    -- Exponent is at position N, take lower 7 bits
    expon = bitCoerce (slice d6 d0 (bytes !! (natToNum @n :: Int))) :: Exponent

-- ============================================================================
-- INTEGRATED WEIGHT MANAGEMENT SYSTEM
-- ============================================================================

data WeightSystemState
  = WSReady          -- Ready for inference
  | WSStreaming      -- Streaming layer from DDR4
  deriving (Generic, NFDataX, Show, Eq)

weightManagementSystem
  :: forall dom . HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Signal dom Bool
  -> Signal dom (Index NumLayers)
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , Signal dom (BitVector 512)
     , Signal dom Bool
     , Signal dom WeightSystemState
     )
weightManagementSystem ddrSlave startStream layerReq sinkReady =
  (ddrMaster, weightStream, streamValid, sysState)
  where
    sysState = register WSReady nextState

    startStreamIfReady = (sysState .==. pure WSReady) .&&. startStream
    (ddrMaster, weightStream, streamValid, streamComplete) =
      layerWeightStreamer ddrSlave layerReq startStreamIfReady sinkReady

    nextState =
      mux (sysState .==. pure WSReady)
        (mux startStreamIfReady (pure WSStreaming) (pure WSReady))
        (mux streamComplete (pure WSReady) (pure WSStreaming))

