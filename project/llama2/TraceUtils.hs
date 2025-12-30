module TraceUtils
  ( -- Basic (no cycle)
    traceChange
  , traceEdge
  , traceWhen
    -- Cycle-aware variants
  , traceChangeC
  , traceEdgeC
  , traceWhenC
  , makeCycleCounter
  ) where

import Clash.Prelude
import Clash.Debug (trace)
import qualified Prelude as P

--------------------------------------------------------------------------------
-- Basic traces (no cycle number)
--------------------------------------------------------------------------------

-- | Trace any signal when its value changes.
traceChange :: (Eq a, Show a, NFDataX a, HiddenClockResetEnable dom)
            => P.String -> Signal dom a -> Signal dom a
traceChange name sig = result
  where
    prev    = register undefined sig
    started = register False (pure True)
    result  = emit <$> started <*> sig <*> prev
    emit False curr _  = curr
    emit True  curr p
      | curr /= p = trace (name P.++ ": " P.++ show p P.++ " -> " P.++ show curr) curr
      | otherwise = curr

-- | Trace Bool signals as RISE/FALL edges.
traceEdge :: (HiddenClockResetEnable dom)
          => P.String -> Signal dom Bool -> Signal dom Bool
traceEdge name sig = result
  where
    prev   = register False sig
    result = emit <$> sig <*> prev
    emit True  False = trace (name P.++ " RISE") True
    emit False True  = trace (name P.++ " FALL") False
    emit curr  _     = curr

-- | Trace a signal whenever a condition is True.
traceWhen :: (Show a)
          => P.String -> Signal dom Bool -> Signal dom a -> Signal dom a
traceWhen name cond sig = result
  where
    result = emit <$> cond <*> sig
    emit True  val = trace (name P.++ ": " P.++ show val) val
    emit False val = val

--------------------------------------------------------------------------------
-- Cycle-aware traces (include cycle number in output)
--------------------------------------------------------------------------------

-- | Create a cycle counter. Call once at top level, pass to *C functions.
makeCycleCounter :: HiddenClockResetEnable dom => Signal dom (Unsigned 32)
makeCycleCounter = cnt
  where cnt = register 0 (cnt + 1)

-- | Trace changes with cycle number: "@123 [tag] sig: old -> new"
traceChangeC :: (Eq a, Show a, NFDataX a, HiddenClockResetEnable dom)
             => Signal dom (Unsigned 32) -> P.String -> Signal dom a -> Signal dom a
traceChangeC cyc name sig = result
  where
    prev    = register undefined sig
    started = register False (pure True)
    result  = emit <$> started <*> cyc <*> sig <*> prev
    emit False _ curr _  = curr
    emit True  c curr p
      | curr /= p = trace ("@" P.++ show c P.++ " " P.++ name P.++ ": " P.++ show p P.++ " -> " P.++ show curr) curr
      | otherwise = curr

-- | Trace edges with cycle number: "@123 [tag] RISE"
traceEdgeC :: (HiddenClockResetEnable dom)
           => Signal dom (Unsigned 32) -> P.String -> Signal dom Bool -> Signal dom Bool
traceEdgeC cyc name sig = result
  where
    prev   = register False sig
    result = emit <$> cyc <*> sig <*> prev
    emit c True  False = trace ("@" P.++ show c P.++ " " P.++ name P.++ " RISE") True
    emit c False True  = trace ("@" P.++ show c P.++ " " P.++ name P.++ " FALL") False
    emit _ curr  _     = curr

-- | Trace on condition with cycle number: "@123 [tag]: value"
traceWhenC :: (Show a)
           => Signal dom (Unsigned 32) -> P.String -> Signal dom Bool -> Signal dom a -> Signal dom a
traceWhenC cyc name cond sig = result
  where
    result = emit <$> cyc <*> cond <*> sig
    emit c True  val = trace ("@" P.++ show c P.++ " " P.++ name P.++ ": " P.++ show val) val
    emit _ False val = val
