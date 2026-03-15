{-# LANGUAGE TemplateHaskell #-}
module LLaMa2.Numeric.FixedPoint
  ( dotProductF
  , rmsNormFwFix
  , invSqrtF
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, Exponent, epsF, scalePow2F)

-- Dot product in FixedPoint
dotProductF :: KnownNat n => Vec n FixedPoint -> Vec n FixedPoint -> FixedPoint
dotProductF a b = sum (zipWith (*) a b)

-- ===========================
-- Fixed-point RMSNorm with inv-sqrt(LUT seed + 1 NR)
-- ===========================

-- 1/sqrt on mantissa in [1,2): 256-entry LUT (compile-time constants).
-- Evaluated at GHC compile time via listToVecTH to avoid integerToDouble# in Clash.
invSqrtMantLUT :: Vec 256 FixedPoint
invSqrtMantLUT = $(listToVecTH
  [ realToFrac (1.0 / sqrt (1.0 + (fromIntegral (k :: Int) + 0.5) / 256.0) :: Double) :: FixedPoint
  | k <- [0..255 :: Int] ])

-- 1/sqrt(2) as direct SFixed literal — no Double intermediate for Clash.
invSqrt2 :: FixedPoint
invSqrt2 = 0.7071067811865476

-- Decompose x = m * 2^e with m in [1,2), using countLeadingZeros.
-- For SFixed 12 20 (= Signed 32): the leading-1 bit at position (31 - clz)
-- represents value 2^((31-clz) - 20) = 2^(11-clz), so e = 11 - clz.
-- This synthesizes to a single LZC circuit rather than a 64-element Vec.
{-# NOINLINE log2Decompose #-}
log2Decompose :: FixedPoint -> (Exponent, FixedPoint)
log2Decompose xIn =
  let x    = max xIn epsF
      clz  = countLeadingZeros (bitCoerce x :: Unsigned 32)
      eBig :: Signed 32
      eBig = 11 - fromIntegral clz          -- range: 11-32 .. 11-0 = -21..11
      e    :: Exponent
      e    = fromIntegral (max (-64 :: Signed 32) (min 63 eBig))
      m    = scalePow2F (negate e) x        -- x * 2^(-e), brings m into [1,2)
  in (e, m)

-- One Newton-Raphson improvement for inv-sqrt
-- Use direct literals to avoid SFixed division (which internally uses SFixed m (m+f)
-- as intermediate type, causing resizeF shiftL(-12) error in Clash synthesis).
nrImproveInvSqrt :: FixedPoint -> FixedPoint -> FixedPoint
nrImproveInvSqrt m y0 =
  let half      = 0.5  -- 1/2 as direct SFixed literal
      threeHalf = 1.5  -- 3/2 as direct SFixed literal
  in y0 * (threeHalf - half * m * y0 * y0)

{-# NOINLINE invSqrtF #-}
invSqrtF :: FixedPoint -> FixedPoint
invSqrtF a0 =
  let a = max a0 epsF
      (e, m) = log2Decompose a
      -- LUT index: top 8 fractional bits of (m-1), m in [1,2) so m-1 in [0,1)
      idx :: Unsigned 8
      idx = floor ((m - 1) * 256 :: FixedPoint)
      seedMant = asyncRom invSqrtMantLUT idx   -- ROM lookup, not mux tree
      -- Even/odd check on exponent using LSB (no Integer needed)
      eIsEven = (e .&. 1) == 0
      -- Scale: 1/sqrt(2^e) = 2^(-e/2).
      -- For even e:   2^(-e/2)   = scalePow2F (-(e `shiftR` 1))
      -- For odd  e:   2^(-(e-1)/2) * invSqrt2
      eHalf    :: Exponent
      eHalf    = shiftR e 1                -- floor(e/2) via arithmetic shift
      eHalfOdd :: Exponent
      eHalfOdd = shiftR (e - 1) 1         -- floor((e-1)/2)
      scalePow :: Exponent
      scalePow = if eIsEven then negate eHalf else negate eHalfOdd
      scale    :: FixedPoint
      scale    = if eIsEven
                 then scalePow2F scalePow 1
                 else scalePow2F scalePow 1 * invSqrt2
      y0 = seedMant * scale
  in nrImproveInvSqrt a y0

{-# NOINLINE rmsNormFwFix #-}
rmsNormFwFix :: forall n. KnownNat n
              => Vec n FixedPoint           -- x
              -> Vec n FixedPoint           -- w (already in FixedPoint)
              -> Vec n FixedPoint
rmsNormFwFix x wF =
  let -- Compute 1/n as a SFixed 12 20 bit-pattern from Int arithmetic.
      -- floor(2^20 / n) gives the reciprocal scaled to fracBits — no Rational needed.
      invNBits :: Signed 32
      invNBits = fromIntegral (div (1048576 :: Int) (natToNum @n :: Int))
      invN     :: FixedPoint
      invN     = bitCoerce invNBits
      meanSq   = sum (map (\xi -> xi*xi) x) * invN + epsF
      invR   = invSqrtF meanSq
      scale  = map (* invR) wF
  in zipWith (*) x scale
