module LLaMa2.Numeric.ParamPack
  ( RowI8E
  , MatI8E
  , quantizeMatI8E
  , dequantRowToF
  ) where

import Clash.Prelude
import LLaMa2.Core.Types (CArray2D(..))
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent, scalePow2F)
import LLaMa2.Numeric.Fixed (quantizeI8E)

-- One row of parameters as (int8 mantissas, shared exponent).
type RowI8E n = (Vec n Mantissa, Exponent)

-- Matrix of rows.
type MatI8E rows cols = Vec rows (RowI8E cols)

-- Elaborate-time quantization: Float -> FixedPoint -> I8E per row.
-- Safe for synthesis because inputs are structural constants.
quantizeMatI8E
  :: ( KnownNat cols)
  =>CArray2D rows cols                 -- Float params baked in the netlist
  -> MatI8E rows cols                 -- Float-free carrier for hardware
quantizeMatI8E (CArray2D rowsF) =
  let rowsFtoF = map (map realToFrac) rowsF
  in map quantizeI8E rowsFtoF

-- Lightweight dequantization to FixedPoint for reuse of F-kernels.
dequantRowToF :: RowI8E n -> Vec n FixedPoint
dequantRowToF (mant, e) =
  let s = scalePow2F e 1
  in map (\q -> fromIntegral q * s) mant
