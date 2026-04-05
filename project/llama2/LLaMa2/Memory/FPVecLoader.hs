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
-- | Extract a single FixedPoint from a 512-bit AXI word.
-- Element i is at bytes [4*i .. 4*i+3], little-endian.
--------------------------------------------------------------------------------
extractFPElem :: BitVector 512 -> Index 16 -> FixedPoint
extractFPElem word idx =
  let i       = fromEnum idx
      byteOff = i * 4
      bytes   = unpack word :: Vec 64 (BitVector 8)
      b0      = bytes !! (fromIntegral (byteOff + 0) :: Index 64)
      b1      = bytes !! (fromIntegral (byteOff + 1) :: Index 64)
      b2      = bytes !! (fromIntegral (byteOff + 2) :: Index 64)
      b3      = bytes !! (fromIntegral (byteOff + 3) :: Index 64)
      bits    :: BitVector 32
      bits    =  resize b0
             .|. (resize b1 `shiftL` 8)
             .|. (resize b2 `shiftL` 16)
             .|. (resize b3 `shiftL` 24)
  in unpack bits

--------------------------------------------------------------------------------
-- | DRAM-backed FixedPoint-vector loader.
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
  -- -----------------------------------------------------------------------
  -- Skid buffer: captures the trigger if the fetcher isn't ready yet
  -- -----------------------------------------------------------------------
  (captureAvail, capturedAddr) =
    Layout.requestCaptureStage fetchTrigger address fetcherReady

  -- -----------------------------------------------------------------------
  -- N-word burst fetcher
  -- -----------------------------------------------------------------------
  (axiMaster, wordsOut, fetchDone, fetcherReady, _dbg, _, _, _) =
    Layout.axiNWordFetcher @dom @(Layout.WordsPerFPVec n)
      dramSlaveIn captureAvail capturedAddr

  -- -----------------------------------------------------------------------
  -- State machine
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

  -- -----------------------------------------------------------------------
  -- Parse on fetch completion
  -- -----------------------------------------------------------------------
  capturing :: Signal dom Bool
  capturing = state .==. pure FPVFetching .&&. fetchDone

  dramVec :: Signal dom (Vec n FixedPoint)
  dramVec = Layout.fixedPointVecParser <$> wordsOut

  -- -----------------------------------------------------------------------
  -- Output register: latch on rising edge of fetchDone
  -- -----------------------------------------------------------------------
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

--------------------------------------------------------------------------------
-- | BRAM-backed FixedPoint-vector loader.
--
-- Like fpVecLoader but stores fetched data in an internal BRAM instead of a
-- Vec register.  Exposes a streaming element-read interface with 1-cycle
-- latency: data at rdAddr[T] is available at rdData[T+1].
--
-- Eliminates the large Vec n FixedPoint output register, replacing it with
-- inference-friendly block RAM.
--------------------------------------------------------------------------------
fpVecLoaderBram :: forall dom n.
  ( HiddenClockResetEnable dom
  , KnownNat n
  , KnownNat (Layout.WordsPerFPVec n)
  )
  => Signal dom (Unsigned 32)         -- ^ cycle counter (unused)
  -> Slave.AxiSlaveIn dom             -- ^ DRAM
  -> Signal dom Bool                  -- ^ fetchTrigger (1-cycle pulse)
  -> Signal dom (Unsigned 32)         -- ^ DRAM address
  -> Signal dom (Index n)             -- ^ rdAddr (element index, pre-issue: 1-cycle latency)
  -> ( Master.AxiMasterOut dom
     , Signal dom FixedPoint          -- ^ rdData (element at rdAddr from previous cycle)
     , Signal dom Bool                -- ^ outputValid
     , Signal dom Bool                -- ^ isBusy
     )
fpVecLoaderBram _cycleCounter dramSlaveIn fetchTrigger address rdAddr =
  (axiMaster, rdData, outputValid, isBusy)
 where
  (captureAvail, capturedAddr) =
    Layout.requestCaptureStage fetchTrigger address fetcherReady

  (axiMaster, _wordsOut, fetchDone, fetcherReady, _dbg, beatWordOut, beatWordValid, beatIdx) =
    Layout.axiNWordFetcher @dom @(Layout.WordsPerFPVec n)
      dramSlaveIn captureAvail capturedAddr

  -- -----------------------------------------------------------------------
  -- BRAM: one 512-bit word per address slot (64 bytes = 16 FixedPoints)
  -- Written word-by-word as AXI beats arrive; read element-by-element.
  -- -----------------------------------------------------------------------

  -- Word address: which 64-byte word within the fetched buffer.
  -- Combinatorial from rdAddr; gives BRAM 1-cycle read latency.
  rdWordAddr :: Signal dom (Index (Layout.WordsPerFPVec n))
  rdWordAddr = (fromIntegral :: Int -> Index (Layout.WordsPerFPVec n)) . (`div` 16) . fromEnum <$> rdAddr

  -- Local index within the 64-byte word (0-15).
  -- Registered to align with 1-cycle BRAM output latency.
  rdLocalIdx :: Signal dom (Index 16)
  rdLocalIdx = register 0 ((fromIntegral :: Int -> Index 16) . (`mod` 16) . fromEnum <$> rdAddr)

  -- Write streaming beats into BRAM as they arrive.
  bramWrOp :: Signal dom (Maybe (Index (Layout.WordsPerFPVec n), BitVector 512))
  bramWrOp = mux beatWordValid
    (Just <$> ((,) <$> beatIdx <*> beatWordOut))
    (pure Nothing)

  -- 1-cycle latency BRAM: read address at T gives output at T+1.
  bramOut :: Signal dom (BitVector 512)
  bramOut = blockRam
    (repeat 0 :: Vec (Layout.WordsPerFPVec n) (BitVector 512))
    rdWordAddr
    bramWrOp

  -- Extract the requested element from the returned word.
  rdData :: Signal dom FixedPoint
  rdData = extractFPElem <$> bramOut <*> rdLocalIdx

  -- -----------------------------------------------------------------------
  -- State machine (same structure as fpVecLoader)
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
