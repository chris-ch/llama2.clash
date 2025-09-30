module Model.Helpers.FixedPoint
  ( dotProductF
  , matrixVectorMultF
  , rmsNormF              -- Float-weighted (back-compat)
  , rmsNormFwFix          -- FixedPoint-weighted (for Q path)
  , invSqrtF
  ) where

import Clash.Prelude
import Model.Core.Types (CArray2D(..))
import Model.Numeric.Types (FixedPoint, ExpS, epsF, scalePow2F)
import Model.Numeric.Fixed (expF)

-- Dot product in FixedPoint
dotProductF :: KnownNat n => Vec n FixedPoint -> Vec n FixedPoint -> FixedPoint
dotProductF a b = sum (zipWith (*) a b)

-- Matrix @ vector where matrix rows are Float params, converted once to FixedPoint.
matrixVectorMultF
  :: forall rows cols. (KnownNat rows, KnownNat cols)
  => CArray2D rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMultF (CArray2D rowsF) xF =
  let rows = map (map realToFrac) rowsF :: Vec rows (Vec cols FixedPoint)
  in map (`dotProductF` xF) rows

-- ===========================
-- Fixed-point RMSNorm with inv-sqrt(LUT seed + 1 NR)
-- ===========================

-- 1/sqrt on mantissa in [1,2): 256-entry LUT (compile-time constants)
invSqrtMantLUT :: Vec 256 FixedPoint
invSqrtMantLUT =
  map
    (\(i :: Index 256) ->
       let m   = 1.0 + (fromIntegral (fromEnum i) + 0.5) / 256.0 :: Double
           val = 1.0 / sqrt m
       in  realToFrac val)
    indicesI

invSqrt2 :: FixedPoint
invSqrt2 = realToFrac (1.0 / sqrt 2.0 :: Double)

-- Decompose x = m * 2^e, with m in [1,2)
log2Decompose :: FixedPoint -> (ExpS, FixedPoint)
log2Decompose xIn =
  let x = max xIn epsF
      pow2 :: Vec 64 FixedPoint
      pow2 = map (\(i :: Index 64) ->
                    let iS = fromInteger (toInteger (fromEnum i) - 32)
                    in scalePow2F iS 1)
                 indicesI
      flags = map (<= x) pow2
      pIdx  = fst (foldl (\(best,seen) (i,b) -> if b then (i,True) else (best,seen))
                         (minBound, False)
                         (zip indicesI flags))
      pInt = toInteger (fromEnum pIdx) - 32
      e    = fromInteger pInt
      m    = x / scalePow2F e 1
  in (e, m)

-- One Newton-Raphson improvement for inv-sqrt
nrImproveInvSqrt :: FixedPoint -> FixedPoint -> FixedPoint
nrImproveInvSqrt m y0 =
  let half = (1 :: FixedPoint) / 2
      threeHalf = (3 :: FixedPoint) / 2
  in y0 * (threeHalf - half * m * y0 * y0)

invSqrtF :: FixedPoint -> FixedPoint
invSqrtF a0 =
  let a = max a0 epsF
      (e, m) = log2Decompose a
      idx :: Unsigned 8
      idx = fromInteger (floor (((m - 1) * 256) :: FixedPoint))
      seedMant = invSqrtMantLUT !! idx
      eInt :: Integer
      eInt = fromIntegral e
      scalePow = if even eInt then negate (fromInteger (eInt `div` 2))
                               else negate (fromInteger ((eInt - 1) `div` 2))
      scale = if even eInt then scalePow2F scalePow 1
                           else scalePow2F scalePow 1 * invSqrt2
      y0 = seedMant * scale
  in nrImproveInvSqrt a y0

-- Variant 1: weights provided as Float (legacy call sites).
rmsNormF :: forall n. KnownNat n
         => Vec n FixedPoint           -- x
         -> Vec n Float                -- w (Float params; converted once)
         -> Vec n FixedPoint
rmsNormF x wFloat =
  let w = map realToFrac wFloat :: Vec n FixedPoint
  in rmsNormFwFix x w

-- Variant 2: weights provided as FixedPoint (Q path).
rmsNormFwFix :: forall n. KnownNat n
              => Vec n FixedPoint           -- x
              -> Vec n FixedPoint           -- w (already in FixedPoint)
              -> Vec n FixedPoint
rmsNormFwFix x wF =
  let n      = fromInteger (natToNum @n) :: FixedPoint
      meanSq = sum (map (\xi -> xi*xi) x) / n + epsF
      invR   = invSqrtF meanSq
      scale  = map (* invR) wF
  in zipWith (*) x scale
