module LLaMa2.Memory.WeightLoader 
(bootWeightLoader, calculateLayerBaseAddress, layerWeightStreamer, parseI8EChunk, weightManagementSystem, WeightSystemState(..), BootLoaderState(..))
    
    where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Types.Parameters
  ( TransformerLayerComponent(..)
  , DecoderParameters(..)
  , MultiHeadAttentionComponentQ(..)
  , FeedForwardNetworkComponentQ(..)
  , SingleHeadComponentQ(..)
  , RotaryEncodingComponentF(..)
  , EmbeddingComponentQ(..)
  )
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import LLaMa2.Numeric.Types (FixedPoint, Exponent, Mantissa)
import LLaMa2.Memory.AxiReadMaster (axiBurstReadMaster)
import LLaMa2.Memory.AxiWriteMaster (axiWriteMaster)
import LLaMa2.Memory.AXI

-- ============================================================================
-- File Layout Constants (matches Parser.hs)
-- ============================================================================

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

-- Calculate total model size in bytes
calculateModelSizeBytes :: Unsigned 32
calculateModelSizeBytes = headerSize + embeddingSize + layersSize + rotarySize
  where
    vocabSize = snatToNum (SNat @VocabularySize)
    modelDim = snatToNum (SNat @ModelDimension)
    numLayers = snatToNum (SNat @NumLayers)
    seqLen = snatToNum (SNat @SequenceLength)
    rotaryDim = snatToNum (SNat @RotaryPositionalEmbeddingDimension)

    headerSize = 7 * 4  -- 7 int32s
    embeddingSize = vocabSize * (modelDim + 1)
    layersSize = numLayers * calculateLayerSizeBytes
    rotarySize = 2 * seqLen * rotaryDim * 4  -- freqCos + freqSin (Float32)

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

-- ============================================================================
-- BOOT LOADER: eMMC → DDR4 (One-time at startup)
-- ============================================================================

data BootLoaderState
  = BootIdle
  | BootReading      -- Reading burst from eMMC
  | BootWriting      -- Writing burst to DDR4
  | BootComplete
  deriving (Generic, NFDataX, Show, Eq)

-- | Boot loader: Copy entire model from eMMC to DDR4
-- Takes ~17.5 seconds for 7GB @ 400 MB/s
bootWeightLoader
  :: forall dom . HiddenClockResetEnable dom
  => AxiSlaveIn dom                     -- eMMC AXI slave
  -> AxiSlaveIn dom                     -- DDR4 AXI slave
  -> Signal dom Bool                    -- Start boot
  -> Signal dom (Unsigned 32)           -- eMMC base address
  -> Signal dom (Unsigned 32)           -- DDR4 base address
  -> ( AxiMasterOut dom                 -- eMMC AXI master
     , AxiMasterOut dom                 -- DDR4 AXI master
     , Signal dom Bool                  -- Boot complete
     , Signal dom (Unsigned 32)         -- Bytes transferred
     , Signal dom BootLoaderState 
     )
bootWeightLoader emmcSlave ddrSlave startBoot emmcBase ddrBase =
  (emmcMaster, ddrMaster, bootComplete, bytesTransferred, state)
  where
    state = register BootIdle nextState

    -- Transfer in 64KB bursts (1024 transfers × 64 bytes)
    burstSize = 1023 :: Unsigned 8  -- AXI len = N-1
    burstBytes = 65536 :: Unsigned 32
    currentBurst = register (0 :: Unsigned 32) nextBurst
    bytesTransferred = currentBurst * pure burstBytes

    -- Total model size
    totalBursts = (calculateModelSizeBytes + burstBytes - 1) `div` burstBytes

    -- Address calculation
    emmcAddr = emmcBase + (currentBurst * pure burstBytes)
    ddrAddr = ddrBase + (currentBurst * pure burstBytes)

    -- Read from eMMC
    startRead = state .==. pure BootReading
    (emmcMaster, readData, readValid, _emmcReady) =
      axiBurstReadMaster emmcSlave emmcAddr (pure burstSize) startRead

    -- Write to DDR4 (stream through)
    startWrite = readValid
    (ddrMaster, writeComplete, ddrReady) =
      axiWriteMaster ddrSlave ddrAddr readData startWrite

    -- Track burst completion
    transfersInBurst = register (0 :: Unsigned 16) nextTransferCount
    burstComplete = transfersInBurst .==. pure 1024

    nextTransferCount = mux startRead
      0
      (mux readValid
        (transfersInBurst + 1)
        transfersInBurst)

    -- All transfers done?
    allComplete = currentBurst .>=. pure totalBursts

    -- State machine
    nextState = mux (state .==. pure BootIdle)
      (mux startBoot (pure BootReading) (pure BootIdle))
      (mux (state .==. pure BootReading)
        (mux (burstComplete .&&. allComplete)
          (pure BootComplete)
          (mux burstComplete
            (pure BootReading)  -- Next burst
            (pure BootReading)))
        (mux (state .==. pure BootComplete)
          (pure BootIdle)
          state))

    nextBurst = mux (state .==. pure BootIdle)
      0
      (mux burstComplete
        (currentBurst + 1)
        currentBurst)

    bootComplete = state .==. pure BootComplete

-- ============================================================================
-- RUNTIME STREAMER: DDR4 → FPGA (Layer-at-a-time)
-- ============================================================================

data StreamerState
  = StreamIdle
  | StreamBursting   -- Streaming bursts
  | StreamComplete
  deriving (Generic, NFDataX, Show, Eq)

-- | Stream one layer's weights from DDR4
-- Takes ~5ms for 200MB @ 42 GB/s
layerWeightStreamer
  :: forall dom . HiddenClockResetEnable dom
  => AxiSlaveIn dom                     -- DDR4 AXI slave
  -> Signal dom (Index NumLayers)       -- Layer to load
  -> Signal dom Bool                    -- Start load
  -> ( AxiMasterOut dom                 -- DDR4 AXI master
     , Signal dom (BitVector 512)       -- Raw weight data (64 bytes)
     , Signal dom Bool                  -- Data valid
     , Signal dom Bool                  -- Load complete
     )
layerWeightStreamer ddrSlave layerIdx startLoad =
  (axiMaster, weightData, dataValid, loadComplete)
  where
    state = register StreamIdle nextState

    -- Calculate layer address
    layerBaseAddr = calculateLayerBaseAddress <$> layerIdx

    -- Burst parameters
    burstSize = 1023 :: Unsigned 8  -- 1024 transfers = 64KB
    burstBytes = 65536 :: Unsigned 32
    totalBursts = (calculateLayerSizeBytes + burstBytes - 1) `div` burstBytes

    currentBurst = register (0 :: Unsigned 32) nextBurst
    burstAddr = layerBaseAddr + (currentBurst * pure burstBytes)

    -- Start bursting
    startBurst = state .==. pure StreamBursting
    (axiMaster, weightData, dataValid, _ready) =
      axiBurstReadMaster ddrSlave burstAddr (pure burstSize) startBurst

    -- Track completion
    transfersInBurst = register (0 :: Unsigned 16) nextTransferCount
    burstComplete = transfersInBurst .==. pure 1024

    nextTransferCount = mux startBurst
      0
      (mux dataValid
        (transfersInBurst + 1)
        transfersInBurst)

    allBurstsComplete = currentBurst .>=. pure totalBursts

    -- State machine
    nextState = mux (state .==. pure StreamIdle)
      (mux startLoad (pure StreamBursting) (pure StreamIdle))
      (mux (state .==. pure StreamBursting)
        (mux (burstComplete .&&. allBurstsComplete)
          (pure StreamComplete)
          (mux burstComplete
            (pure StreamBursting)  -- Next burst
            (pure StreamBursting)))
        (mux (state .==. pure StreamComplete)
          (pure StreamIdle)
          state))

    nextBurst = mux (state .==. pure StreamIdle)
      0
      (mux burstComplete
        (currentBurst + 1)
        currentBurst)

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
  = WSBoot           -- Boot: loading eMMC → DDR4
  | WSReady          -- Ready for inference
  | WSStreaming      -- Streaming layer from DDR4
  deriving (Generic, NFDataX, Show, Eq)

-- | Complete weight management system
-- Find the weightManagementSystem function and ADD a bypass parameter:

weightManagementSystem
  :: forall dom . HiddenClockResetEnable dom
  => Signal dom Bool                    -- NEW: bypass (skip boot for simulation)
  -> AxiSlaveIn dom                     -- eMMC AXI slave
  -> AxiSlaveIn dom                     -- DDR4 AXI slave
  -> Signal dom Bool                    -- Power on / start boot
  -> Signal dom (Index NumLayers)       -- Current layer
  -> Signal dom Bool                    -- Load layer trigger
  -> ( AxiMasterOut dom                 -- eMMC AXI master
     , AxiMasterOut dom                 -- DDR4 AXI master
     , Signal dom (BitVector 512)       -- Weight data stream
     , Signal dom Bool                  -- Weight data valid
     , Signal dom Bool                  -- System ready
     , Signal dom (Unsigned 32)         -- Boot progress (bytes)
     , Signal dom WeightSystemState     -- expose state
     , Signal dom BootLoaderState  -- boot state
     )
weightManagementSystem bypass emmcSlave ddrSlave powerOn layerRequest loadTrigger =
  (emmcMaster, ddrMaster, weightStream, streamValid, systemReady, bootProgress, sysState, bootState)
  where
    -- When bypassed, immediately mark as ready
    systemReady' = mux bypass 
      (pure True)                       -- Bypass: always ready
      (sysState .==. pure WSReady)     -- Normal: wait for boot
    systemReady = sysState .==. pure WSReady

    sysState = register WSBoot nextSysState
    
    emmcBaseAddr = 0 :: Unsigned 32
    ddrBaseAddr = 0 :: Unsigned 32
    
    startBoot = powerOn .&&. (sysState .==. pure WSBoot) .&&. (not <$> bypass)
    (emmcMaster, ddrMasterBoot, bootDone, bootProgress, bootState) =
      bootWeightLoader emmcSlave ddrSlave startBoot
                       (pure emmcBaseAddr) (pure ddrBaseAddr)
    
    startStream = (sysState .==. pure WSReady) .&&. loadTrigger .&&. (not <$> bypass)
    (ddrMasterRuntime, weightStream, streamValid, streamComplete) =
      layerWeightStreamer ddrSlave layerRequest startStream
    
    isBoot = sysState .==. pure WSBoot
    ddrMaster = AxiMasterOut
      { arvalid = mux bypass (pure False) (mux isBoot (arvalid ddrMasterBoot) (arvalid ddrMasterRuntime))
      , ardata  = mux bypass (pure (errorX "bypass")) (mux isBoot (ardata ddrMasterBoot) (ardata ddrMasterRuntime))
      , rready  = mux bypass (pure False) (mux isBoot (rready ddrMasterBoot) (rready ddrMasterRuntime))
      , awvalid = mux bypass (pure False) (mux isBoot (awvalid ddrMasterBoot) (awvalid ddrMasterRuntime))
      , awdata  = mux bypass (pure (errorX "bypass")) (mux isBoot (awdata ddrMasterBoot) (awdata ddrMasterRuntime))
      , wvalid  = mux bypass (pure False) (mux isBoot (wvalid ddrMasterBoot) (wvalid ddrMasterRuntime))
      , wdataMI = mux bypass (pure (errorX "bypass")) (mux isBoot (wdataMI ddrMasterBoot) (wdataMI ddrMasterRuntime))
      , bready  = mux bypass (pure False) (mux isBoot (bready ddrMasterBoot) (bready ddrMasterRuntime))
      }
    
    nextSysState = mux bypass
      (pure WSReady)  -- Bypass: skip directly to ready
      (mux (sysState .==. pure WSBoot)
        (mux bootDone (pure WSReady) (pure WSBoot))
        (mux (sysState .==. pure WSReady)
          (mux startStream (pure WSStreaming) (pure WSReady))
          (mux (sysState .==. pure WSStreaming)
            (mux streamComplete (pure WSReady) (pure WSStreaming))
            sysState)))
