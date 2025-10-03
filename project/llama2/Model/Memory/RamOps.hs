module Model.Memory.RamOps (
    RamOp(..),
    toRamOperation,
    runTdpRamOps
) where

import Clash.Prelude
import Data.Maybe (isJust)

-- Convert separate read and optional-write signals into a unified RAM operation stream.
toRamOperation
  :: NFDataX a
  => Signal dom (Index n)              -- ^ read address
  -> Signal dom (Maybe (Index n, a))   -- ^ optional write
  -> Signal dom (RamOp n a)
toRamOperation rdAddr wrM =
  mux (isJust <$> wrM)
      (uncurry RamWrite . fromJustX <$> wrM)
      (RamRead <$> rdAddr)

-- Run a true-dual-port BRAM from two RamOp streams.
-- Initializes memory to zero.
runTdpRamOps
  :: forall dom n a
   . ( HiddenClockResetEnable dom
     , KnownNat n
     , NFDataX a )
  => Signal dom (RamOp n a)  -- ^ Port A op stream
  -> Signal dom (RamOp n a)  -- ^ Port B op stream
  -> ( Signal dom a          -- ^ Port A read data
     , Signal dom a )        -- ^ Port B read data
runTdpRamOps opA opB = (qA, qB)
 where
  (qA, qB) = trueDualPortBlockRam opA opB
