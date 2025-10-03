module Model.Numeric.Fixed
  ( quantizeI8E
  , expF
  ) where

import Clash.Prelude
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
            qf x = satRoundToI8 (round (x * s))
            rec x = fromIntegral (qf x) * scalePow2F e 1
        in sum (map (\x -> let d = x - rec x in d*d) xs)
      eBest = if errFor eFloor <= errFor eCeil then eFloor else eCeil
      k = scalePow2F (negate eBest) 1
      qElem x =
        let y  = x * k
            yr = if y >= 0 then floor (y + 0.5) else ceiling (y - 0.5) :: Integer
        in satRoundToI8 yr
  in (map qElem xs, eBest)

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

