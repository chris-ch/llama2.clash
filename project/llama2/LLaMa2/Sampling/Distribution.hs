module LLaMa2.Sampling.Distribution (
    pickSample, uniformRandom01Generator
) where

import Clash.Prelude

import Data.Maybe (fromMaybe)
import LLaMa2.Types.LayerData ( Token)

import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (expF)

-- xorshift32 unchanged
xorshift32 :: Unsigned 32 -> Unsigned 32
xorshift32 s0 =
  let s1 = s0 `xor` shiftL s0 13
      s2 = s1 `xor` shiftR s1 17
      s3 = s2 `xor` shiftL s2 5
  in s3

pickSample :: (KnownNat n, KnownNat (n + 1)) => FixedPoint -> Vec (n + 1) FixedPoint -> FixedPoint -> Token
pickSample temperature logits rand = if temperature <= 0 then argMax logits
        else let probabilities = softmax temperature logits
             in categoricalSampler rand probabilities

pseudoRandomGenerator :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool           -- ^ readyPulse
  -> Signal dom (Unsigned 32)  -- ^ seed
  -> Signal dom (Unsigned 32)  -- ^ prng state/output
pseudoRandomGenerator readyPulse seedSig =
  mealyB step 0 (readyPulse, seedSig)
  where
    step :: Unsigned 32 -> (Bool, Unsigned 32) -> (Unsigned 32, Unsigned 32)
    step s (rdy, seedNow) =
      let s' = if rdy then xorshift32 (seedNow `xor` 0x9E3779B9) else xorshift32 s
      in  (s', s')

uniformRandom01Generator :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom FixedPoint
uniformRandom01Generator readyPulse seed =
  (/ realToFrac (16777216.0 :: Double)) . fromIntegral . (`shiftR` 8)
    <$> pseudoRandomGenerator readyPulse seed

-- categorical sampling from FixedPoint probabilities
categoricalSampler :: forall n. (KnownNat (n + 1), KnownNat n) => FixedPoint -> Vec (n + 1) FixedPoint -> Unsigned 32
categoricalSampler u probs =
  let cdf = scanl1 (+) probs
      idxM = findIndex (>= u) cdf
  in fromIntegral (fromEnum (fromMaybe maxBound idxM))

softmax :: forall n. KnownNat (n + 1) => FixedPoint -> Vec (n + 1) FixedPoint -> Vec (n + 1) FixedPoint
softmax t xs =
  let m    = maximum xs
      exps = map (\x -> expF ((x - m) / t)) xs
      s    = sum exps
  in map (/ s) exps

-- argmax (returns Token)
argMax :: forall n. (KnownNat (n + 1)) => Vec (n+1) FixedPoint -> Unsigned 32
argMax vec =
  let (ix, _) = foldl (\(iBest, vBest) (i, v) -> if v > vBest then (i, v) else (iBest, vBest))
                      (0, head vec)
                      (imap (\i x -> (fromIntegral i, x)) vec)
  in ix
