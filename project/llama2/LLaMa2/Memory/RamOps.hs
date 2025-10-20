module LLaMa2.Memory.RamOps (
    trueDualPortRam
) where

import Clash.Prelude
import Data.Maybe (isJust)

-- Convert separate read and optional-write signals into a unified RAM operation stream.
ramOpConverter
  :: NFDataX a
  => Signal dom (Index n)              -- ^ read address
  -> Signal dom (Maybe (Index n, a))   -- ^ optional write
  -> Signal dom (RamOp n a)
ramOpConverter rdAddr wrM =
  mux (isJust <$> wrM)
      (uncurry RamWrite . fromJustX <$> wrM)
      (RamRead <$> rdAddr)

-- True-dual-port BRAM runner using standard Clash interface.
-- Initializes memory to zero.
trueDualPortRam
  :: forall dom n a
    . ( HiddenClockResetEnable dom
      , KnownNat n
      , NFDataX a )
  => Signal dom (Index n)              -- Port A read address
  -> Signal dom (Maybe (Index n, a))   -- Port A optional write
  -> Signal dom (Index n)              -- Port B read address
  -> Signal dom (Maybe (Index n, a))   -- Port B optional write
  -> ( Signal dom a                    -- Port A read data
      , Signal dom a )                 -- Port B read data
trueDualPortRam rdAddrA wrM_A rdAddrB wrM_B =
  trueDualPortBlockRam
    (ramOpConverter rdAddrA wrM_A)
    (ramOpConverter rdAddrB wrM_B)
