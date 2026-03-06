module LLaMa2.Memory.FPVecLoader
  ( fpVecLoader
  ) where

import Clash.Prelude
import qualified Prelude as P

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
--
-- Cross-checks the DRAM result against the HC reference and calls P.error on
-- any mismatch.
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
  -> Vec n FixedPoint                 -- ^ HC reference for cross-check
  -> String                           -- ^ tag for error messages
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec n FixedPoint)  -- ^ output (holds last fetched value)
     , Signal dom Bool                -- ^ outputValid
     , Signal dom Bool                -- ^ isBusy
     )
fpVecLoader _cycleCounter dramSlaveIn fetchTrigger address hcVec tag =
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
    mux newFetchStarting                                        (pure FPVFetching) $
    mux (state .==. pure FPVFetching .&&. fetchDone)           (pure FPVReady)    $
    state

  -- -----------------------------------------------------------------------
  -- Parse and cross-check on fetch completion
  -- -----------------------------------------------------------------------
  capturing :: Signal dom Bool
  capturing = state .==. pure FPVFetching .&&. fetchDone

  dramVec :: Signal dom (Vec n FixedPoint)
  dramVec = Layout.fixedPointVecParser <$> wordsOut

  checkedVec :: Signal dom (Vec n FixedPoint)
  checkedVec = checkPure <$> dramVec
   where
    checkPure dv =
      let pairs      = P.zip [0..] (P.zip (toList dv) (toList hcVec))
          mismatches = P.filter (\(_, (d, h)) -> d P./= h) pairs
      in if P.null mismatches then dv
         else let (i, (d, h)) = P.head mismatches
              in P.error $ tag P.++ "DRAM/HC mismatch"
                        P.++ ": index " P.++ show (i :: Int)
                        P.++ " DRAM=" P.++ show d P.++ " HC=" P.++ show h

  -- -----------------------------------------------------------------------
  -- Output register: latch on rising edge of fetchDone
  -- -----------------------------------------------------------------------
  outputVec :: Signal dom (Vec n FixedPoint)
  outputVec = register (repeat 0) $ mux capturing checkedVec outputVec

  outputValid :: Signal dom Bool
  outputValid = state .==. pure FPVReady .&&. (not <$> newFetchStarting)

  isBusy :: Signal dom Bool
  isBusy = state .==. pure FPVFetching .||. newFetchStarting
