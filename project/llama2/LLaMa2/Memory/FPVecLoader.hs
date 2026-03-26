module LLaMa2.Memory.FPVecLoader
  ( fpVecLoader
  , fpVecLoaderDyn
  , fpVecLoaderBram
  ) where

import Clash.Prelude

import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout

--------------------------------------------------------------------------------
-- State machine
--------------------------------------------------------------------------------
data FPVState = FPVIdle | FPVFetching | FPVReady
  deriving (Generic, NFDataX, Show, Eq)

--------------------------------------------------------------------------------
-- | DRAM-backed FixedPoint-vector loader (Vec output).
--
-- Fetches Vec n FixedPoint from DRAM as a burst of WordsPerFPVec n 64-byte
-- words.  After completion the result is held in an output register until the
-- next fetch trigger.
--------------------------------------------------------------------------------
fpVecLoader :: forall dom n.
  ( HiddenClockResetEnable dom
  , KnownNat n
  , KnownNat (Layout.WordsPerFPVec n)
  )
  => Signal dom (Unsigned 32)         -- ^ cycle counter (unused, kept for API symmetry)
  -> Slave.AxiSlaveIn dom             -- ^ DRAM
  -> Signal dom Bool                  -- ^ fetchTrigger (1-cycle pulse)
  -> Signal dom (Unsigned 32)         -- ^ DRAM address
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec n FixedPoint)  -- ^ output (holds last fetched value)
     , Signal dom Bool                -- ^ outputValid
     , Signal dom Bool                -- ^ isBusy
     )
fpVecLoader _cycleCounter dramSlaveIn fetchTrigger address =
  (axiMaster, outputVec, outputValid, isBusy)
 where
  (captureAvail, capturedAddr) =
    Layout.requestCaptureStage fetchTrigger address fetcherReady

  (axiMaster, wordsOut, fetchDone, fetcherReady, _dbg, _, _, _) =
    Layout.axiNWordFetcher @dom @(Layout.WordsPerFPVec n)
      dramSlaveIn captureAvail capturedAddr

  state :: Signal dom FPVState
  state = register FPVIdle nextState

  newFetchStarting :: Signal dom Bool
  newFetchStarting =
    captureAvail .&&. fetcherReady .&&. (state ./=. pure FPVFetching)

  nextState :: Signal dom FPVState
  nextState =
    mux newFetchStarting (pure FPVFetching) $
    mux (state .==. pure FPVFetching .&&. fetchDone) (pure FPVReady) state

  capturing :: Signal dom Bool
  capturing = state .==. pure FPVFetching .&&. fetchDone

  dramVec :: Signal dom (Vec n FixedPoint)
  dramVec = Layout.fixedPointVecParser <$> wordsOut

  outputVec :: Signal dom (Vec n FixedPoint)
  outputVec = register (repeat 0) $ mux capturing dramVec outputVec

  outputValid :: Signal dom Bool
  outputValid = state .==. pure FPVReady .&&. (not <$> newFetchStarting)

  isBusy :: Signal dom Bool
  isBusy = state .==. pure FPVFetching .||. newFetchStarting

-- | Variant of fpVecLoader with a dynamic address signal.
-- Use when the address varies at runtime (e.g., per-position rotary tables).
fpVecLoaderDyn :: forall dom n.
  ( HiddenClockResetEnable dom
  , KnownNat n
  , KnownNat (Layout.WordsPerFPVec n)
  )
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom Bool                    -- ^ fetchTrigger (1-cycle pulse)
  -> Signal dom (Unsigned 32)           -- ^ DRAM address
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec n FixedPoint)
     , Signal dom Bool                  -- ^ outputValid
     , Signal dom Bool                  -- ^ isBusy
     )
fpVecLoaderDyn _cycleCounter dramSlaveIn fetchTrigger address =
  (axiMaster, outputVec, outputValid, isBusy)
 where
  (captureAvail, capturedAddr) =
    Layout.requestCaptureStage fetchTrigger address fetcherReady

  (axiMaster, wordsOut, fetchDone, fetcherReady, _dbg, _, _, _) =
    Layout.axiNWordFetcher @dom @(Layout.WordsPerFPVec n)
      dramSlaveIn captureAvail capturedAddr

  state :: Signal dom FPVState
  state = register FPVIdle nextState

  newFetchStarting :: Signal dom Bool
  newFetchStarting =
    captureAvail .&&. fetcherReady .&&. (state ./=. pure FPVFetching)

  nextState :: Signal dom FPVState
  nextState =
    mux newFetchStarting (pure FPVFetching) $
    mux (state .==. pure FPVFetching .&&. fetchDone) (pure FPVReady) state

  capturing :: Signal dom Bool
  capturing = state .==. pure FPVFetching .&&. fetchDone

  dramVec :: Signal dom (Vec n FixedPoint)
  dramVec = Layout.fixedPointVecParser <$> wordsOut

  outputVec :: Signal dom (Vec n FixedPoint)
  outputVec = register (repeat 0) $ mux capturing dramVec outputVec

  outputValid :: Signal dom Bool
  outputValid = state .==. pure FPVReady .&&. (not <$> newFetchStarting)

  isBusy :: Signal dom Bool
  isBusy = state .==. pure FPVFetching .||. newFetchStarting

-- | BRAM-backed FP-vector loader.
--
-- Instead of accumulating the burst into a Vec register, each AXI beat is
-- written directly to an internal blockRam as it arrives. The caller supplies
-- a read address and receives the corresponding FixedPoint element with
-- 1-cycle BRAM latency. This eliminates the large Vec output register.
--
-- Element layout: each 512-bit AXI word holds 16 × 32-bit FixedPoint values.
-- BRAM depth = WordsPerFPVec n (e.g. 48 for n = 768).
-- Read address is the element index (0..n-1); the function computes the
-- word address and the local-within-word index internally.
fpVecLoaderBram :: forall dom n.
  ( HiddenClockResetEnable dom
  , KnownNat n
  , KnownNat (Layout.WordsPerFPVec n)
  )
  => Signal dom (Unsigned 32)         -- ^ cycle counter (unused)
  -> Slave.AxiSlaveIn dom             -- ^ DRAM
  -> Signal dom Bool                  -- ^ fetchTrigger (1-cycle pulse)
  -> Signal dom (Unsigned 32)         -- ^ DRAM address
  -> Signal dom (Index n)             -- ^ rdAddr: element index to read
  -> ( Master.AxiMasterOut dom
     , Signal dom FixedPoint          -- ^ rdData: element at rdAddr (1-cycle latency)
     , Signal dom Bool                -- ^ outputValid (fetch complete, level)
     , Signal dom Bool                -- ^ isBusy
     )
fpVecLoaderBram _cycleCounter dramSlaveIn fetchTrigger address rdAddr =
  (axiMaster, rdData, outputValid, isBusy)
 where
  (captureAvail, capturedAddr) =
    Layout.requestCaptureStage fetchTrigger address fetcherReady

  -- Streaming beat outputs: write each beat to BRAM as it arrives.
  (axiMaster, _, fetchDone, fetcherReady, _, beatWordOut, beatWordValid, beatIdx) =
    Layout.axiNWordFetcher @dom @(Layout.WordsPerFPVec n)
      dramSlaveIn captureAvail capturedAddr

  -- -----------------------------------------------------------------------
  -- Internal BRAM: depth = WordsPerFPVec n, width = BitVector 512.
  -- Write port: driven by streaming beats (one 512-bit word per AXI beat).
  -- Read port: driven by word address derived from rdAddr.
  -- -----------------------------------------------------------------------
  -- Word address: element index >> 4 (each 512-bit word holds 16 elements)
  rdWordAddr :: Signal dom (Index (Layout.WordsPerFPVec n))
  rdWordAddr = (fromIntegral . (`shiftR` 4) . (fromIntegral :: Index n -> Unsigned 32)) <$> rdAddr

  bramOut :: Signal dom (BitVector 512)
  bramOut = blockRam (repeat 0 :: Vec (Layout.WordsPerFPVec n) (BitVector 512))
              rdWordAddr
              (mux beatWordValid
                (Just <$> liftA2 (,) beatIdx beatWordOut)
                (pure Nothing))

  -- Local index within the 512-bit word, registered by 1 cycle to align with BRAM output.
  rdLocalIdx :: Signal dom (Index 16)
  rdLocalIdx = register 0 $
    (fromIntegral . (.&. 0xF) . (fromIntegral :: Index n -> Unsigned 32)) <$> rdAddr

  -- Extract one FixedPoint element from a 512-bit word using the same
  -- little-endian byte reassembly as fixedPointVecParser:
  --   element i occupies bytes [4i .. 4i+3] of the 512-bit word
  --   (byte 0 of each element is the LSByte, matching memory layout)
  extractElem :: BitVector 512 -> Index 16 -> FixedPoint
  extractElem w localIdx =
    let bytes   = unpack w :: Vec 64 (BitVector 8)
        byteOff = (fromIntegral localIdx :: Unsigned 7) * 4
        b0 = bytes !! (fromIntegral byteOff       :: Index 64)
        b1 = bytes !! (fromIntegral (byteOff + 1) :: Index 64)
        b2 = bytes !! (fromIntegral (byteOff + 2) :: Index 64)
        b3 = bytes !! (fromIntegral (byteOff + 3) :: Index 64)
        bits :: BitVector 32
        bits =   resize b0
             .|. (resize b1 `shiftL` 8)
             .|. (resize b2 `shiftL` 16)
             .|. (resize b3 `shiftL` 24)
    in unpack bits

  rdData :: Signal dom FixedPoint
  rdData = extractElem <$> bramOut <*> rdLocalIdx

  -- -----------------------------------------------------------------------
  -- State machine (identical to fpVecLoader)
  -- -----------------------------------------------------------------------
  state :: Signal dom FPVState
  state = register FPVIdle nextState

  newFetchStarting :: Signal dom Bool
  newFetchStarting =
    captureAvail .&&. fetcherReady .&&. (state ./=. pure FPVFetching)

  nextState :: Signal dom FPVState
  nextState =
    mux newFetchStarting (pure FPVFetching) $
    mux (state .==. pure FPVFetching .&&. fetchDone) (pure FPVReady) state

  outputValid :: Signal dom Bool
  outputValid = state .==. pure FPVReady .&&. (not <$> newFetchStarting)

  isBusy :: Signal dom Bool
  isBusy = state .==. pure FPVFetching .||. newFetchStarting
