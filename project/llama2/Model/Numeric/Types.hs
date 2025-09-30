module Model.Numeric.Types
  ( F
  , fracBitsF
  , intBitsF
  , epsF
  , I8E(..)
  , ExpS
  , Act
  , Wgt
  , Acc
  , satRoundToI8
  , scalePow2F
  , clampExp
  ) where

import Clash.Prelude
import GHC.Generics (Generic)

-- Fixed-point scalar: range ~[-2048,2048), 20 fractional bits.
type F = SFixed 12 20
intBitsF, fracBitsF :: Int
intBitsF  = 12
fracBitsF = 20
epsF :: F
epsF = 2 ^^ negate fracBitsF

type Act  = Signed 8
type Wgt  = Signed 8
type Acc  = Signed 32
type ExpS = Signed 7   -- clamp to [-64,63]

data I8E = I8E { mant :: Signed 8, expo :: ExpS }
  deriving (Show, Eq, Generic, NFDataX)

satRoundToI8 :: Integer -> Signed 8
satRoundToI8 x =
  let lo = -127; hi = 127
      y | x < lo = lo
        | x > hi = hi
        | otherwise = x
  in fromInteger y

-- Multiply by 2^n (n signed), fully synthesizable.
scalePow2F :: ExpS -> F -> F
scalePow2F n x =
  let nInt = fromIntegral n :: Integer
  in if nInt >= 0
        then x *  fromInteger (1 `shiftL` fromInteger nInt)
        else x / fromInteger (1 `shiftL` fromInteger (negate nInt))

clampExp :: ExpS -> ExpS
clampExp e = max (-64) (min 63 e)
