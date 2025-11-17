{-# LANGUAGE DerivingVia #-}
module LLaMa2.Numeric.Quantization
  ( expF
  , dequantRowToF
  , RowI8E (..)
  , MatI8E
  ) where
import Clash.Prelude
import LLaMa2.Numeric.Types

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

-- 2^f with f in [0,1); LUT index stays in Unsigned 8 (no Integer on datapath)
exp2Frac :: FixedPoint -> FixedPoint
exp2Frac f =
  let fClamped = max 0 (min (1 - epsF) f)
      idx :: Unsigned 8
      idx = floor (fClamped * 256)  -- floor to Unsigned 8
  in exp2FracLUT !! idx

-- expF: x -> 2^(x/ln2) = 2^n * 2^f, all bounded ints on datapath
expF :: FixedPoint -> FixedPoint
expF x =
  let y  = x * ln2InvF
      nC :: Exponent
      nC = clampExp (floor y)
      f  = y - fromIntegral nC
      b  = exp2Frac f
  in scalePow2F nC b

-- One row of parameters as (int8 mantissas, shared exponent).
data RowI8E n = RowI8E { rowMantissas :: Vec n Mantissa, rowExponent :: Exponent}
  deriving stock (Eq, Show, Generic)
  deriving NFDataX via (RowI8E n)

-- Matrix of rows.
type MatI8E rows cols = Vec rows (RowI8E cols)

-- Lightweight dequantization to FixedPoint for reuse of F-kernels.
dequantRowToF :: RowI8E n -> Vec n FixedPoint
dequantRowToF (RowI8E {rowMantissas = mant, rowExponent = e}) =
  let s = scalePow2F e 1
  in map (\q -> fromIntegral q * s) mant
