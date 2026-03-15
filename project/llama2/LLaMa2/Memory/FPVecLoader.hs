module LLaMa2.Memory.FPVecLoader
  ( fpVecLoader
  , fpVecLoaderDyn
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
  (axiMaster, wordsOut, fetchDone, fetcherReady, _dbg) =
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

  (axiMaster, wordsOut, fetchDone, fetcherReady, _dbg) =
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
