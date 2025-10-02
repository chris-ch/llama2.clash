module Model.Numeric.Fixed
  ( quantizeI8E
  , quantizeI8E_ceilSafe
  , expF
  ) where

import Clash.Prelude
import GHC.Generics (Generic)
import Model.Numeric.Types (FixedPoint, Activation, Exponent, scalePow2F, clampExp, satRoundToI8, epsF)

-- ===========================
-- Quantization: F <-> I8E (PoT)
-- ===========================

-- Quantize a vector to Signed 8 mantissas with a shared Signed 7 exponent.
-- Nearest-PoT exponent (reduces MSE vs floor-PoT)
-- No Floating is used to find the exponent: we compare against a ROM of 2^i.
quantizeI8E :: forall n. KnownNat n => Vec n FixedPoint -> (Vec n Activation, Exponent)
quantizeI8E xs =
  let maxAbs :: FixedPoint
      maxAbs = foldl max 0 (map abs xs)
      pow2 :: Vec 64 FixedPoint
      pow2 = map (\(i :: Index 64) ->
                    let iS :: Exponent
                        iS = fromInteger (toInteger (fromEnum i) - 32)
                    in scalePow2F iS 1)
                 indicesI
      flags :: Vec 64 Bool
      flags = map (<= maxAbs) pow2
      pIdx :: Index 64
      pIdx = fst (foldl (\(best, seen) (i,b) -> if b then (i,True) else (best,seen))
                        (minBound, False)
                        (zip indicesI flags))
      pInt :: Integer
      pInt = toInteger (fromEnum pIdx) - 32
      eFloor :: Exponent
      eFloor = clampExp (fromInteger (pInt - 7))
      eCeil  :: Exponent
      eCeil  = clampExp (fromInteger (pInt - 6))
      errFor :: Exponent -> FixedPoint
      errFor e =
        let s  = scalePow2F (negate e) 1
            qf x = fromIntegral (satRoundToI8 (round (x * s)))
            rec x = fromIntegral (qf x) * scalePow2F e 1
        in sum (map (\x -> let d = x - rec x in d*d) xs)
      eBest = if errFor eFloor <= errFor eCeil then eFloor else eCeil
      k = scalePow2F (negate eBest) 1
      qElem x =
        let y  = x * k
            yr = if y >= 0 then floor (y + 0.5) else ceiling (y - 0.5) :: Integer
        in satRoundToI8 yr
  in (map qElem xs, eBest)

-- Ceil-safe exponent: guarantees no clipping, i.e., |q| <= 127 for all elements
quantizeI8E_ceilSafe :: forall n. KnownNat n => Vec n FixedPoint -> (Vec n Activation, Exponent)
quantizeI8E_ceilSafe xs =
  let maxAbs :: FixedPoint
      maxAbs = foldl max 0 (map abs xs)
      -- Find integer p such that 2^p <= maxAbs < 2^(p+1)
      pow2 :: Vec 64 (Exponent, FixedPoint)
      pow2 = map (\(i :: Index 64) ->
                    let e :: Exponent = fromInteger (toInteger (fromEnum i) - 32)
                    in (e, scalePow2F e 1))
                 indicesI
      (pE, pV) =
        foldl
          (\(bestE, bestV) (e, v) -> if v <= maxAbs && v > bestV then (e, v) else (bestE, bestV))
          (minBound, 0)
          pow2
      -- If maxAbs <= 127 * 2^p, we can use e = p-7; else e = p-6
      th = 127 * pV
      eRaw :: Exponent
      eRaw = if maxAbs <= th then clampExp (pE - 7) else clampExp (pE - 6)
      sInv = scalePow2F (negate eRaw) 1
      qElem x =
        let y  = x * sInv
            yr = if y >= 0 then floor (y + 0.5) else ceiling (y - 0.5) :: Integer
        in satRoundToI8 yr
  in (map qElem xs, eRaw)

-- Nearest power-of-two exponent for a positive FixedPoint a.
-- Searches e in [-64 .. 63] and returns the argmin_e |2^e - a|.
nearestPow2Exp :: FixedPoint -> Exponent
nearestPow2Exp aIn =
  let a = max epsF (abs aIn)  -- ensure positive, non-zero
      pow2Vec :: Vec 128 (Exponent, FixedPoint)
      pow2Vec =
        map
          (\(i :: Index 128) ->
             let e :: Exponent
                 e = fromInteger (toInteger (fromEnum i) - 64)
             in  (e, scalePow2F e 1))
          indicesI
      -- initialize with first element and fold the tail
      (e0, v0) = head pow2Vec
      d0       = abs (v0 - a)
      pickBest (bestE, bestV, bestD) (e, v) =
        let d = abs (v - a)
        in if d < bestD then (e, v, d) else (bestE, bestV, bestD)
      (bestE, _, _) = foldl pickBest (e0, v0, d0) (tail pow2Vec)
  in bestE

-- Round x/s to nearest integer (symmetric), saturate to int8 [-127,127].
qElemWithScale :: FixedPoint -> FixedPoint -> Activation
qElemWithScale s x =
  let y  = x / s
      yr = if y >= 0 then floor (y + 0.5) else ceiling (y - 0.5) :: Integer
  in satRoundToI8 yr

dequantizeI8E :: (Vec n Activation, Exponent) -> Vec n FixedPoint
dequantizeI8E (qs, e) =
  let s = scalePow2F e 1
  in map (\q -> fromIntegral q * s) qs

-- ===========================
-- expF using 2^x decomposition with LUT-256
-- ===========================

ln2InvF :: FixedPoint
ln2InvF = realToFrac (1.4426950408889634 :: Double)  -- 1/ln(2)

-- 256-entry ROM for 2^(k/256), k=0..255
exp2FracLUT :: Vec 256 FixedPoint
exp2FracLUT =
  map
    (\(i :: Index 256) ->
       let k   = fromIntegral (fromEnum i) :: Double
           val = 2 ** (k / 256)
       in  realToFrac val)
    indicesI

-- 2^f with f in [0,1); nearest-neighbor LUT
exp2Frac :: FixedPoint -> FixedPoint
exp2Frac f =
  let fClamped = max 0 (min (1 - epsF) f)
      idx :: Unsigned 8
      idx = fromInteger (floor (fClamped * 256))  -- 0..255
  in exp2FracLUT !! idx

-- expF: x -> 2^(x/ln2) = 2^n * 2^f
expF :: FixedPoint -> FixedPoint
expF x =
  let y  = x * ln2InvF
      nI = floor y :: Integer
      f  = y - fromInteger nI
      b  = exp2Frac f
      nC :: Exponent
      nC = clampExp (fromInteger nI)
  in scalePow2F nC b

expF' :: FixedPoint -> FixedPoint
expF' x =
  let ln2InvF = realToFrac (1.4426950408889634 :: Double)
      exp2FracLUT :: Vec 256 FixedPoint
      exp2FracLUT = map (\(i :: Index 256) -> realToFrac (2 ** (fromIntegral (fromEnum i) / 256 :: Double))) indicesI
      exp2Frac f =
        let idx :: Unsigned 8 = fromInteger (floor (max 0 (min (1 - (2 ^^ (-20))) f) * 256))
        in exp2FracLUT !! idx
      y  = x * ln2InvF
      nI = floor y :: Integer
      f  = y - fromInteger nI
      b  = exp2Frac f
      nC :: Exponent = clampExp (fromInteger nI)
  in scalePow2F nC b
