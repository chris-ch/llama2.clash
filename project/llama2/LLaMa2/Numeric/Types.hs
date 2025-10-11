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

-- Multiply by 2^n (n signed), fully synthesizable.
scalePow2F :: Exponent -> FixedPoint -> FixedPoint
scalePow2F n x =
  let nInt = fromIntegral n :: Integer
  in if nInt >= 0
        then x *  fromInteger (1 `shiftL` fromIntegral nInt)
        else x / fromInteger (1 `shiftL` fromIntegral (negate nInt))

clampExp :: Exponent -> Exponent
clampExp e = max (-64) (min 63 e)
