module LLaMa2.Numeric.RmsNormSeq
  ( rmsNormSeq
  , rmsNormSeqVec
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
  | RNNormalize  -- ^ Pass 2: write x[i] * scale * w[i] into xHat BRAM
  | RNDone       -- ^ outputValid held high; waits for next validIn
  deriving (Generic, NFDataX, Show, Eq)

-- | Sequential RMS normalisation with BRAM-backed output.
--
-- Replaces the fully-combinational 'rmsNormFwFix' with a state machine that
-- processes one element per cycle, avoiding the wide combinational cloud that
-- causes Vivado elaboration to run out of memory for large model dimensions.
--
-- Both xi and wi are supplied element-by-element from BRAMs (1-cycle latency).
-- 'rdNext' is the pre-issue address: BRAM[T] receives rdNext[T], delivering
-- xi[counter] and wi[counter] at cycle T+1 when counter = rdNext[T-1].
--
-- On a one-cycle 'validIn' pulse:
--
-- @
--   RNIdle      (waiting)
--   RNAccum     (n cycles)  acc += x[i]^2;  rdNext = counter+1
--   RNScale     (1 cycle)   scale = invSqrtF(acc/n + eps)
--   RNNormalize (n cycles)  xHatWrite = Just (i, x[i] * scale * w[i])
--   RNDone      (held)      outputValid = True
-- @
--
-- Total latency: 2n + 2 cycles from 'validIn' to 'outputValid'.
-- 'outputValid' is a level signal held until the next 'validIn'.
--
-- The caller must create an xHat BRAM driven by 'xHatWrite', and feed the
-- BRAM read data back as 'xi' on the next cycle (using 'rdNext' as read addr).
-- Similarly 'wi' must come from a BRAM (e.g. fpVecLoaderBram) at 'rdNext'.
{-# NOINLINE rmsNormSeq #-}
rmsNormSeq
  :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool               -- ^ validIn: one-cycle pulse
  -> Signal dom FixedPoint         -- ^ xi (element at rdNext address, 1-cycle BRAM latency)
  -> Signal dom FixedPoint         -- ^ wi (weight at rdNext address, 1-cycle BRAM latency)
  -> ( Signal dom Bool                             -- ^ outputValid (level, held from RNDone)
     , Signal dom (Maybe (Index n, FixedPoint))    -- ^ xHatWrite (write to xHat BRAM each RNNormalize cycle)
     , Signal dom (Index n)                        -- ^ counter  (current element index)
     , Signal dom (Index n)                        -- ^ rdNext   (next element index; pre-issue to BRAMs)
     )
rmsNormSeq validIn xi wi = (outputValid, xHatWrite, counter, nextCounter)
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

    state   = register RNIdle  nextState
    counter = register 0       nextCounter
    acc     = register 0       nextAcc
    scale   = register 0       nextScale

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

    -- ----------------------------------------------------------------
    -- Normalised element: x[counter] * scale * w[counter]
    -- Both xi and wi arrive from BRAMs with 1-cycle latency aligned to counter.
    -- ----------------------------------------------------------------
    outElem :: Signal dom FixedPoint
    outElem = (\x_ s w_ -> x_ * s * w_) <$> xi <*> scale <*> wi

    -- Write one normalised element per cycle during RNNormalize.
    xHatWrite :: Signal dom (Maybe (Index n, FixedPoint))
    xHatWrite = mux inNormalize
      (Just <$> ((,) <$> counter <*> outElem))
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

-- | Vec-based variant of rmsNormSeq for use where the full output Vec is
-- needed (e.g. the attention QKV path).  Weights are supplied as a Vec
-- (loaded once from DRAM) and the normalised result is accumulated into an
-- output Vec register, one element per cycle.
--
-- Interface identical to the original rmsNormSeq before the BRAM refactor.
{-# NOINLINE rmsNormSeqVec #-}
rmsNormSeqVec
  :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool               -- ^ validIn: one-cycle pulse
  -> Signal dom FixedPoint         -- ^ xi (element at counter; 1-cycle BRAM latency, driven by rdNext)
  -> Signal dom (Vec n FixedPoint) -- ^ w  (weight vector, stable throughout)
  -> ( Signal dom Bool               -- ^ outputValid
     , Signal dom (Vec n FixedPoint) -- ^ result
     , Signal dom (Index n)          -- ^ counter
     , Signal dom (Index n)          -- ^ rdNext
     )
rmsNormSeqVec validIn xi wSig = (outputValid, outReg, counter, nextCounter)
  where
    invNBits :: Signed 32
    invNBits = fromIntegral (div (1048576 :: Int) (natToNum @n :: Int))
    invN :: FixedPoint
    invN = bitCoerce invNBits

    state   :: Signal dom RmsNormState
    counter :: Signal dom (Index n)
    acc     :: Signal dom FixedPoint
    scale   :: Signal dom FixedPoint
    outReg  :: Signal dom (Vec n FixedPoint)

    state   = register RNIdle     nextState
    counter = register 0          nextCounter
    acc     = register 0          nextAcc
    scale   = register 0          nextScale
    outReg  = register (repeat 0) nextOutReg

    wi :: Signal dom FixedPoint
    wi = (!!) <$> wSig <*> counter

    inAccum     = (== RNAccum)     <$> state
    inScale     = (== RNScale)     <$> state
    inNormalize = (== RNNormalize) <$> state
    inDone      = (== RNDone)      <$> state
    inIdle      = (== RNIdle)      <$> state

    atMax = (== maxBound) <$> counter

    nextCounter :: Signal dom (Index n)
    nextCounter =
      mux (inAccum .||. inNormalize) ((+ 1) <$> counter) (pure 0)

    nextAcc :: Signal dom FixedPoint
    nextAcc =
      mux inAccum ((\a x_ -> a + x_ * x_) <$> acc <*> xi) $
      mux (validIn .&&. (inIdle .||. inDone)) (pure 0) acc

    nextScale :: Signal dom FixedPoint
    nextScale =
      mux inScale
        (invSqrtF . (+ epsF) <$> ((*) <$> acc <*> pure invN))
        scale

    outElem :: Signal dom FixedPoint
    outElem = (\x_ s w_ -> x_ * s * w_) <$> xi <*> scale <*> wi

    nextOutReg :: Signal dom (Vec n FixedPoint)
    nextOutReg =
      mux inNormalize (replace <$> counter <*> outElem <*> outReg) outReg

    nextState :: Signal dom RmsNormState
    nextState =
      mux (inIdle .&&. validIn)    (pure RNAccum)     $
      mux (inAccum .&&. atMax)     (pure RNScale)     $
      mux inScale                  (pure RNNormalize) $
      mux (inNormalize .&&. atMax) (pure RNDone)      $
      mux (inDone .&&. validIn)    (pure RNAccum)
      state

    outputValid :: Signal dom Bool
    outputValid = inDone
