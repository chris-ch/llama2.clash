module LLaMa2.Numeric.Types
  ( FixedPoint
  , fracBitsF
  , intBitsF
  , epsF
  , I8E(..)
  , Exponent
  , Mantissa
  , Activation
  , Weight
  , Accumulator
  , satRoundToI8
  , scalePow2F
  , clampExp
  ) where

import Clash.Prelude

-- Fixed-point scalar: range ~[-2048,2048), 20 fractional bits.
type FixedPoint = SFixed 12 20
intBitsF, fracBitsF :: Int
intBitsF  = 12
fracBitsF = 20
epsF :: FixedPoint
epsF = 2 ^^ negate fracBitsF

type Activation = Signed 8
type Weight  = Signed 8
type Accumulator  = Signed 32
type Exponent = Signed 7   -- clamp to [-64,63]
type Mantissa = Signed 8


data I8E = I8E { mantissa :: Mantissa, exponent :: Exponent }
  deriving (Show, Eq, Generic, NFDataX)

satRoundToI8 :: Integer -> Signed 8
satRoundToI8 x =
  let lo = -127; hi = 127
      y | x < lo = lo
        | x > hi = hi
        | otherwise = x
  in fromInteger y

-- 128-entry ROM: pow2LUT !! k = 2^(k-64) as FixedPoint.
-- Evaluated entirely at GHC compile time via listToVecTH to avoid
-- Clash synthesis of Double arithmetic (integerToDouble# blackbox).
pow2LUT :: Vec 128 FixedPoint
pow2LUT = $(listToVecTH
  [ (realToFrac :: Double -> SFixed 12 20) (2.0 ** (fromIntegral (k :: Int) - 64.0))
  | k <- [0..127 :: Int] ])

-- Multiply by 2^n using a ROM lookup. n is clamped to [-64,63] by the caller.
-- Uses asyncRom (Clash primitive) instead of Vec.!! to avoid mux-tree expansion.
{-# NOINLINE scalePow2F #-}
scalePow2F :: Exponent -> FixedPoint -> FixedPoint
scalePow2F n x = x * asyncRom pow2LUT idx
  where
    idx :: Index 128
    idx = fromIntegral (resize n + (64 :: Signed 8))

clampExp :: Exponent -> Exponent
clampExp e = max (-64) (min 63 e)
