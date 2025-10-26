module LLaMa2.Memory.WeightLoader.BootWeightLoader
  ( bootWeightLoader, BootLoaderState(..), calculateLayerSizeBytes, calculateModelSizeBytes
  ) where

import Clash.Prelude
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Types.ModelConfig (VocabularySize, ModelDimension, NumLayers, SequenceLength, RotaryPositionalEmbeddingDimension, HiddenDimension, NumQueryHeads, NumKeyValueHeads, HeadDimension)
import qualified LLaMa2.Memory.AxiReadMaster as ReadMaster (axiBurstReadMaster)
import qualified LLaMa2.Memory.AxiWriteMaster as WriteMaster (axiWriteMaster)


-- ============================================================================
-- BOOT LOADER: eMMC → DDR4 (One-time at startup)
-- ============================================================================

data BootLoaderState
  = BootIdle
  | BootReading
  | BootPause1      -- NEW: First pause cycle
  | BootPause2      -- NEW: Second pause cycle  
  | BootComplete
  deriving (Generic, NFDataX, Show, Eq)


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
     , Signal dom Bool                  -- readValid (for debugging)
     , Signal dom Bool                  -- writerDataReady (for debugging)
     , Signal dom (Unsigned 8)         -- transfersInBurst (for debugging)
     , Signal dom Bool                  -- burstComplete (for debugging)
     , Signal dom Bool                  -- allComplete (for debugging)
     , Signal dom (Unsigned 32)        -- currentBurst (for debugging)
     , Signal dom Bool                  -- burstStarted (for debugging)
     , Signal dom Bool                  -- startReadBurst (for debugging)
     )
bootWeightLoader emmcSlave ddrSlave startBoot emmcBase ddrBase =
  (emmcMaster, ddrMaster, bootComplete, bytesTransferred, state, readValid, 
    writerDataReady, transfersInBurst, burstComplete, allComplete,
    currentBurst, burstStarted, needsStart)
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
    emmcAddr = register 0 $ emmcBase + (currentBurst * pure burstBytes)
    ddrAddr  = register 0 $ ddrBase + (currentBurst * pure burstBytes)

    -- Only start when master is explicitly ready:
    needsStart = (state .==. pure BootReading) .&&. 
                emmcReady .&&.  -- NEW
                (not <$> burstStarted) .&&.
                (transfersInBurst .==. pure 0)

    -- If axiBurstReadMaster provides a "ready" output:
    ( emmcMaster, readData, readValid, emmcReady ) = 
      ReadMaster.axiBurstReadMaster emmcSlave emmcAddr (pure burstLen) needsStart writerDataReady

    -- Reader: start one burst when entering BootReading
    -- Use burstStarted flag to track if current burst already initiated (more reliable than pulse1)
    burstStarted = register False nextBurstStarted
    nextBurstStarted =
      mux (state .==. pure BootPause2) (pure False) $
      mux needsStart (pure True) burstStarted

    -- Writer: use same logic to start write burst
    ( ddrMaster
      , _writeDone
      , writerDataReady ) = WriteMaster.axiWriteMaster ddrSlave ddrAddr (pure burstLen) needsStart readData readValid

    -- Count accepted beats within this burst (readValid == writer accepted beat, because we wire writerDataReady back)
    transfersInBurst = register (0 :: Unsigned 8) nextTransferCount

    burstComplete =
      (state .==. pure BootReading) .&&.
      (transfersInBurst .==. pure burstLen)

    nextTransferCount =
      mux (state ./=. pure BootReading) 0 $
      mux (transfersInBurst .>=. pure (burstLen - 1)) (pure burstLen) $
      mux readValid (transfersInBurst + 1) transfersInBurst

    -- Progress state machine: Reading -> Pause (to bump burst index) -> Complete/Reading
    allComplete = (currentBurst + 1) .>=. pure totalBursts

    nextState =
      mux (state .==. pure BootIdle)
        (mux startBoot (pure BootReading) (pure BootIdle)) $
      mux (state .==. pure BootReading)
        (mux burstComplete (pure BootPause1) (pure BootReading)) $
      mux (state .==. pure BootPause1)
        (pure BootPause2) $  -- Always transition
      mux (state .==. pure BootPause2)
        (mux allComplete (pure BootComplete) (pure BootReading)) $
      pure BootIdle

    nextBurst =
      mux (state .==. pure BootIdle) 0 $
      mux (state .==. pure BootPause1) (currentBurst + 1)
      currentBurst

    bootComplete = state .==. pure BootComplete
