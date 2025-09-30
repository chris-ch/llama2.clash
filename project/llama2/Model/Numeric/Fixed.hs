-- ===== project/llama2/Model/Numeric/Fixed.hs =====
module Model.Numeric.Fixed
  ( quantizeI8E
  , dequantizeI8E
  , expF
  , ln2InvF
  ) where

import Clash.Prelude
import GHC.Generics (Generic)
import Model.Numeric.Types

-- ===========================
-- Quantization: F <-> I8E (PoT)
-- ===========================

-- Quantize a vector to Signed 8 mantissas with a shared Signed 7 exponent.
-- We choose e = floor(log2(maxAbs)) - 7 so that magnitudes roughly fit in [-127,127].
-- No Floating is used to find the exponent: we compare against a ROM of 2^i.
quantizeI8E :: forall n. KnownNat n => Vec n F -> (Vec n Act, ExpS)
quantizeI8E xs =
  let maxAbs :: F
      maxAbs = foldl max 0 (map abs xs)

      -- 2^i for i in [-32 .. 31]
      pow2 :: Vec 64 F
      pow2 = map (\(i :: Index 64) ->
                    let iS :: ExpS
                        iS = fromInteger (toInteger (fromEnum i) - 32)
                    in scalePow2F iS 1)
                 indicesI

      -- flags[i] = True if 2^i <= maxAbs
      flags :: Vec 64 Bool
      flags = map (<= maxAbs) pow2

      -- last True index (or 0 if all False)
      pIdx :: Index 64
      pIdx = fst (foldl
                    (\(best, seen) (i,b) -> if b then (i,True) else (best,seen))
                    (minBound, False)
                    (zip indicesI flags))

      pInt :: Integer
      pInt = toInteger (fromEnum pIdx) - 32

      e :: ExpS
      e = clampExp (fromInteger (pInt - 7))

      k :: F  -- 2^-e
      k = scalePow2F (negate e) 1

      qElem :: F -> Act
      qElem x =
        let y  = x * k
            yr = if y >= 0 then floor (y + 0.5) else ceiling (y - 0.5) :: Integer
        in satRoundToI8 yr
  in (map qElem xs, e)

dequantizeI8E :: (Vec n Act, ExpS) -> Vec n F
dequantizeI8E (qs, e) =
  let s = scalePow2F e 1
  in map (\q -> fromIntegral q * s) qs

-- ===========================
-- expF using 2^x decomposition with LUT-256
-- ===========================

ln2InvF :: F
ln2InvF = realToFrac (1.4426950408889634 :: Double)  -- 1/ln(2)

-- 256-entry ROM for 2^(k/256), k=0..255
exp2FracLUT :: Vec 256 F
exp2FracLUT =
  map
    (\(i :: Index 256) ->
       let k   = fromIntegral (fromEnum i) :: Double
           val = 2 ** (k / 256)
       in  realToFrac val)
    indicesI

-- 2^f with f in [0,1); nearest-neighbor LUT
exp2Frac :: F -> F
exp2Frac f =
  let fClamped = max 0 (min (1 - epsF) f)
      idx :: Unsigned 8
      idx = fromInteger (floor (fClamped * 256))  -- 0..255
  in exp2FracLUT !! idx

-- expF: x -> 2^(x/ln2) = 2^n * 2^f
expF :: F -> F
expF x =
  let y  = x * ln2InvF
      nI = floor y :: Integer
      f  = y - fromInteger nI
      b  = exp2Frac f
      nC :: ExpS
      nC = clampExp (fromInteger nI)
  in scalePow2F nC b
