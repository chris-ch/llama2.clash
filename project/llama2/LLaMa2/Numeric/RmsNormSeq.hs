module LLaMa2.Numeric.RmsNormSeq
  ( rmsNormSeq
  , RmsNormState(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, epsF)
import LLaMa2.Numeric.FixedPoint (invSqrtF)

-- | State for the sequential RMS-norm state machine.
data RmsNormState
  = RNIdle
  | RNAccum      -- ^ Pass 1: accumulate sum(x[i]^2), one element per cycle
  | RNScale      -- ^ 1 cycle: compute scale = invSqrtF(acc/n + eps)
  | RNNormalize  -- ^ Pass 2: write x[i] * scale * w[i] into output BRAM
  | RNDone       -- ^ outputValid held high; waits for next validIn
  deriving (Generic, NFDataX, Show, Eq)

-- | Sequential RMS normalisation.
--
-- Replaces the fully-combinational 'rmsNormFwFix' with a state machine that
-- processes one element per cycle, avoiding the wide Vec registers that cause
-- Vivado elaboration OOM for large model dimensions.
--
-- Both 'xi' and 'wi' are scalar BRAM outputs delivered with 1-cycle latency.
-- The caller drives the x-BRAM and w-BRAM read ports with 'rdNext', which is
-- the next counter value (issued 1 cycle before it is needed).
--
-- On a one-cycle 'validIn' pulse the machine proceeds:
--
-- @
--   RNIdle      (waiting)
--   RNAccum     (n cycles)  acc += x[i]^2
--   RNScale     (1 cycle)   scale = invSqrtF(acc/n + eps)
--   RNNormalize (n cycles)  bramWrite = Just (i, x[i] * scale * w[i])
--   RNDone      (held)      outputValid = True
-- @
--
-- Total latency: 2n + 2 cycles from 'validIn' to first 'outputValid'.
-- 'outputValid' is a level signal held until the next 'validIn'.
--
-- Hardware cost: 2 DSP multipliers + 1 adder (all scalar, registered).
-- Output is written element-by-element to the caller's BRAM via 'bramWrite'.
{-# NOINLINE rmsNormSeq #-}
rmsNormSeq
  :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool               -- ^ validIn: one-cycle pulse
  -> Signal dom FixedPoint         -- ^ xi: x element at 'counter' (from x-BRAM, 1-cycle latency)
  -> Signal dom FixedPoint         -- ^ wi: weight element at 'counter' (from w-BRAM, 1-cycle latency)
  -> ( Signal dom Bool                           -- ^ outputValid (level, held from RNDone)
     , Signal dom (Maybe (Index n, FixedPoint))  -- ^ bramWrite: xHat BRAM write port
     , Signal dom (Index n)                      -- ^ counter: current element index
     , Signal dom (Index n)                      -- ^ rdNext: pre-fetch address for x- and w-BRAMs
     )
rmsNormSeq validIn xi wi = (outputValid, bramWrite, counter, nextCounter)
  where
    -- Compile-time reciprocal: floor(2^20 / n) as FixedPoint bit pattern.
    invNBits :: Signed 32
    invNBits = fromIntegral (div (1048576 :: Int) (natToNum @n :: Int))
    invN :: FixedPoint
    invN = bitCoerce invNBits

    -- ----------------------------------------------------------------
    -- State registers
    -- ----------------------------------------------------------------
    state   :: Signal dom RmsNormState
    counter :: Signal dom (Index n)
    acc     :: Signal dom FixedPoint
    scale   :: Signal dom FixedPoint

    state   = register RNIdle nextState
    counter = register 0      nextCounter
    acc     = register 0      nextAcc
    scale   = register 0      nextScale

    -- ----------------------------------------------------------------
    -- Phase predicates
    -- ----------------------------------------------------------------
    inAccum     :: Signal dom Bool
    inAccum     = (== RNAccum)     <$> state
    inScale     :: Signal dom Bool
    inScale     = (== RNScale)     <$> state
    inNormalize :: Signal dom Bool
    inNormalize = (== RNNormalize) <$> state
    inDone      :: Signal dom Bool
    inDone      = (== RNDone)      <$> state
    inIdle      :: Signal dom Bool
    inIdle      = (== RNIdle)      <$> state

    -- Counter wraps at maxBound (Index n arithmetic is modular)
    atMax :: Signal dom Bool
    atMax = (== maxBound) <$> counter

    -- ----------------------------------------------------------------
    -- Next-state logic
    -- ----------------------------------------------------------------
    nextCounter :: Signal dom (Index n)
    nextCounter =
      mux (inAccum .||. inNormalize)
        ((+ 1) <$> counter)   -- wraps to 0 at maxBound
        (pure 0)

    -- Accumulate during RNAccum; reset to 0 when starting a new run
    nextAcc :: Signal dom FixedPoint
    nextAcc =
      mux inAccum
        ((\a x_ -> a + x_ * x_) <$> acc <*> xi)
      $ mux (validIn .&&. (inIdle .||. inDone))
        (pure 0)
        acc

    -- Latch invSqrtF result during the single RNScale cycle
    nextScale :: Signal dom FixedPoint
    nextScale =
      mux inScale
        (invSqrtF . (+ epsF) <$> ((*) <$> acc <*> pure invN))
        scale

    -- One normalised element per cycle during RNNormalize.
    outElem :: Signal dom FixedPoint
    outElem = (\x_ s w_ -> x_ * s * w_) <$> xi <*> scale <*> wi

    -- BRAM write port: valid during RNNormalize only.
    bramWrite :: Signal dom (Maybe (Index n, FixedPoint))
    bramWrite = mux inNormalize
      ((\c e -> Just (c, e)) <$> counter <*> outElem)
      (pure Nothing)

    nextState :: Signal dom RmsNormState
    nextState =
      mux (inIdle .&&. validIn)          (pure RNAccum)     $
      mux (inAccum .&&. atMax)           (pure RNScale)     $
      mux inScale                        (pure RNNormalize) $
      mux (inNormalize .&&. atMax)       (pure RNDone)      $
      mux (inDone .&&. validIn)          (pure RNAccum)
      state

    outputValid :: Signal dom Bool
    outputValid = inDone
