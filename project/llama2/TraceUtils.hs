module TraceUtils
  ( traceChangeC
  , traceEdgeC
  , traceWhenC
  , makeCycleCounter
  ) where

import Clash.Prelude
import qualified Prelude as P

--------------------------------------------------------------------------------
-- Cycle-aware traces (include cycle number in output)
--------------------------------------------------------------------------------

-- | Create a cycle counter. Call once at top level, pass to *C functions.
makeCycleCounter :: forall dom . HiddenClockResetEnable dom => Signal dom (Unsigned 32)
makeCycleCounter = cnt
  where cnt = register 0 (cnt + 1)

-- | Trace changes with cycle number: "@123 [tag] sig: old -> new"
-- (No-op for simulation performance)
traceChangeC :: (Eq a, Show a, NFDataX a, HiddenClockResetEnable dom)
             => Signal dom (Unsigned 32) -> P.String -> Signal dom a -> Signal dom a
traceChangeC _cyc _name sig = sig

-- | Trace edges with cycle number: "@123 [tag] RISE"
-- (No-op for simulation performance; re-enable by restoring trace calls)
traceEdgeC :: (HiddenClockResetEnable dom)
           => Signal dom (Unsigned 32) -> P.String -> Signal dom Bool -> Signal dom Bool
traceEdgeC _cyc _name sig = sig

-- | Trace on condition with cycle number: "@123 [tag]: value"
-- (No-op for simulation performance)
traceWhenC :: Signal dom (Unsigned 32) -> P.String -> Signal dom Bool -> Signal dom a -> Signal dom a
traceWhenC _cyc _name _cond sig = sig
