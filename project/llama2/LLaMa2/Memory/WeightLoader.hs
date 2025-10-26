module LLaMa2.Memory.WeightLoader 
 (
  bootWeightLoader,
  calculateModelSizeBytes,
  calculateLayerSizeBytes,
  calculateLayerBaseAddress,
  layerWeightStreamer,
  parseI8EChunk,
  weightManagementSystem,
  WeightSystemState(..), BootLoaderState(..))
    
    where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Numeric.Types (Exponent, Mantissa)
import LLaMa2.Memory.AxiWriteMaster (axiWriteMaster)
import LLaMa2.Memory.AxiReadMaster (axiBurstReadMaster)
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn)
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut (..))

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
  | BootPause        --
  | BootWriting      -- Writing burst to DDR4
  | BootComplete
  deriving (Generic, NFDataX, Show, Eq)

-- | Boot loader: eMMC -> DDR4, burst-to-burst streaming (read master Ready-aware, write master burst-capable)
bootWeightLoader :: forall dom . HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom                     -- eMMC AXI slave
  -> Slave.AxiSlaveIn dom                     -- DDR4 AXI slave
  -> Signal dom Bool                    -- Start boot
  -> Signal dom (Unsigned 32)           -- eMMC base address
  -> Signal dom (Unsigned 32)           -- DDR4 base address
  -> ( Master.AxiMasterOut dom                 -- eMMC AXI master
     , Master.AxiMasterOut dom                 -- DDR4 AXI master
     , Signal dom Bool                  -- Boot complete
     , Signal dom (Unsigned 32)         -- Bytes transferred (approx: completed bursts * 16KB)
     , Signal dom BootLoaderState
     )
bootWeightLoader emmcSlave ddrSlave startBoot emmcBase ddrBase =
  (emmcMaster, ddrMaster, bootComplete, bytesTransferred, state)
  where
    state = register BootIdle nextState

    -- 256 beats/burst, 64 bytes/beat
    burstLen   = 255 :: Unsigned 8
    burstBytes = 16384 :: Unsigned 32

    -- Burst index and “bytes transferred” counter (coarse, per burst)
    currentBurst     = register (0 :: Unsigned 32) nextBurst
    bytesTransferred = currentBurst * pure burstBytes

    totalBursts = (calculateModelSizeBytes + burstBytes - 1) `div` burstBytes

    -- Addresses for the current burst
    emmcAddr = emmcBase + (currentBurst * pure burstBytes)
    ddrAddr  = ddrBase  + (currentBurst * pure burstBytes)

    -- Reader: start one burst while in BootReading
    startReadBurst = state .==. pure BootReading
    ( emmcMaster
      , readData
      , readValid
      , _emmcReady     -- goes High in ReadIdle; not used further
      ) = axiBurstReadMaster emmcSlave emmcAddr (pure burstLen) startReadBurst writerDataReady

    -- Writer: start one burst when we enter BootReading and the per-burst counter is 0
    atBurstStart   = (state .==. pure BootReading) .&&. (transfersInBurst .==. pure 0)
    startWriteBurst= pulse1 atBurstStart
    ( ddrMaster
      , _writeDone
      , writerDataReady ) = axiWriteMaster ddrSlave ddrAddr (pure burstLen) startWriteBurst readData readValid

    -- Count accepted beats within this burst (readValid == writer accepted beat, because we wire writerDataReady back)
    transfersInBurst = register (0 :: Unsigned 16) nextTransferCount
    burstComplete    = transfersInBurst .==. pure 256
    nextTransferCount =
      mux (state ./=. pure BootReading) 0 $
      mux burstComplete 0 $
      mux readValid (transfersInBurst + 1) transfersInBurst

    -- Progress state machine: Reading -> Pause (to bump burst index) -> Complete/Reading
    allComplete = currentBurst .>=. pure totalBursts

    nextState =
      mux (state .==. pure BootIdle)
        (mux startBoot (pure BootReading) (pure BootIdle)) $
      mux (state .==. pure BootReading)
        (mux burstComplete (pure BootPause) (pure BootReading)) $
      mux (state .==. pure BootPause)
        (mux allComplete (pure BootComplete) (pure BootReading)) $
      -- BootComplete self-clears next cycle (optional)
      pure BootIdle

    nextBurst =
      mux (state .==. pure BootIdle) 0 $
      mux ((state .==. pure BootReading) .&&. burstComplete) (currentBurst + 1) currentBurst

    bootComplete = state .==. pure BootComplete

    -- One-shot pulse helper
    pulse1 :: HiddenClockResetEnable dom => Signal dom Bool -> Signal dom Bool
    pulse1 s = s .&&. not <$> register False s

-- ============================================================================
-- RUNTIME STREAMER: DDR4 → FPGA (Layer-at-a-time)
-- ============================================================================

data StreamerState
  = StreamIdle
  | StreamBursting   -- Streaming bursts
  | StreamComplete
  deriving (Generic, NFDataX, Show, Eq)

-- | Stream one layer's weights from DDR4 (READY-aware) — counter bug fixed
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
    (axiMaster, weightData, dataValid, _ready) =
      axiBurstReadMaster ddrSlave burstAddr (pure burstLen) startBurst sinkReady

    transfersInBurst = register (0 :: Unsigned 16) nextTransferCount
    burstComplete    = transfersInBurst .==. pure 256

    -- BUGFIX: do not reset while in StreamBursting each cycle; reset only on Idle or when a burst completes
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
  = WSBoot           -- Boot: loading eMMC → DDR4
  | WSReady          -- Ready for inference
  | WSStreaming      -- Streaming layer from DDR4
  deriving (Generic, NFDataX, Show, Eq)

-- | Complete weight management system
-- Find the weightManagementSystem function and ADD a bypass parameter:
-- Select between two AXI master records (record-of-signals) using a Signal Bool
selectAxiMasterOut
  :: Signal dom Bool         -- ^ sel=True => pick 'a', otherwise 'b'
  -> Master.AxiMasterOut dom        -- ^ a
  -> Master.AxiMasterOut dom        -- ^ b
  -> Master.AxiMasterOut dom
selectAxiMasterOut sel a b = Master.AxiMasterOut
  { arvalid = mux sel (Master.arvalid a) (Master.arvalid b)
  , ardata  = mux sel (Master.ardata  a) (Master.ardata  b)
  , rready  = mux sel (Master.rready  a) (Master.rready  b)
  , awvalid = mux sel (Master.awvalid a) (Master.awvalid b)
  , awdata  = mux sel (Master.awdata  a) (Master.awdata  b)
  , wvalid  = mux sel (Master.wvalid  a) (Master.wvalid  b)
  , wdata = mux sel (Master.wdata a) (Master.wdata b)
  , bready  = mux sel (Master.bready  a) (Master.bready  b)
  }

weightManagementSystem
  :: forall dom . HiddenClockResetEnable dom
  => Signal dom Bool                    -- bypass
  -> Slave.AxiSlaveIn dom                     -- eMMC AXI slave
  -> Slave.AxiSlaveIn dom                     -- DDR4 AXI slave
  -> Signal dom Bool                    -- Power on / start boot
  -> Signal dom (Index NumLayers)       -- Current layer
  -> Signal dom Bool                    -- Load layer trigger
  -> Signal dom Bool                    -- streamSinkReady (consumer ready for beats)
  -> ( Master.AxiMasterOut dom
     , Master.AxiMasterOut dom
     , Signal dom (BitVector 512)
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom (Unsigned 32)
     , Signal dom WeightSystemState
     , Signal dom BootLoaderState
     )
weightManagementSystem bypassBoot emmcSlave ddrSlave powerOn layerRequest loadTrigger streamSinkReady =
  (emmcMaster, ddrMaster, weightStream, streamValid, systemReadyOut, bootProgress, sysState, bootState)
  where
    sysState = register WSBoot nextSysState
    emmcBaseAddr = 0 :: Unsigned 32
    ddrBaseAddr  = 0 :: Unsigned 32

    startBoot = powerOn .&&. (sysState .==. pure WSBoot) .&&. (not <$> bypassBoot)
    (emmcMasterBoot, ddrMasterBoot, bootDone, bootProgress, bootState) =
      bootWeightLoader emmcSlave ddrSlave startBoot (pure emmcBaseAddr) (pure ddrBaseAddr)

    startStream = (sysState .==. pure WSReady) .&&. loadTrigger
    (ddrMasterRuntime, weightStream, streamValid, streamComplete) =
      layerWeightStreamer ddrSlave layerRequest startStream streamSinkReady

    -- An idle/zeroed AXI master for eMMC when bypassing boot
    emmcIdle = Master.AxiMasterOut
      { arvalid = pure False, ardata  = pure (errorX "emmc bypass")
      , rready  = pure False
      , awvalid = pure False, awdata  = pure (errorX "emmc bypass")
      , wvalid  = pure False, wdata = pure (errorX "emmc bypass")
      , bready  = pure False
      }

    -- eMMC master: pick idle when bypass=True, else boot master
    emmcMaster = selectAxiMasterOut bypassBoot emmcIdle emmcMasterBoot

    -- DDR master: pick BOOT path only when NOT bypassing AND in WSBoot; otherwise Runtime
    isBoot  = sysState .==. pure WSBoot
    useBoot = (not <$> bypassBoot) .&&. isBoot
    ddrMaster = selectAxiMasterOut useBoot ddrMasterBoot ddrMasterRuntime

    nextSysState =
      mux bypassBoot
        (pure WSReady)
        (mux (sysState .==. pure WSBoot)
             (mux bootDone (pure WSReady) (pure WSBoot))
             (mux (sysState .==. pure WSReady)
                  (mux startStream (pure WSStreaming) (pure WSReady))
                  (mux (sysState .==. pure WSStreaming)
                       (mux streamComplete (pure WSReady) (pure WSStreaming))
                       sysState)))

    systemReadyOut = mux bypassBoot (pure True) (sysState .==. pure WSReady)
