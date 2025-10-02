module Model.Numeric.ParamPack
  ( RowI8E
  , MatI8E
  , QArray2D(..)
  , quantizeMatI8E
  , dequantRowToF
  ) where

import Clash.Prelude
import Model.Core.Types (CArray2D(..))
import Model.Numeric.Types (FixedPoint, Activation, Exponent, scalePow2F)
import Model.Numeric.Fixed (quantizeI8E)

-- One row of parameters as (int8 mantissas, shared exponent).
type RowI8E n = (Vec n Activation, Exponent)

-- Matrix of rows.
type MatI8E rows cols = Vec rows (RowI8E cols)

-- Named wrapper for readability at component boundaries.
newtype QArray2D (rows :: Nat) (cols :: Nat) =
  QArray2D { unQ2D :: MatI8E rows cols }
  deriving (Generic, Show, Eq)

instance NFDataX (MatI8E rows cols) => NFDataX (QArray2D rows cols) where
  deepErrorX :: String -> QArray2D rows cols
  deepErrorX s = QArray2D (deepErrorX s)
  rnfX ::QArray2D rows cols -> ()
  rnfX (QArray2D m) = rnfX m

-- Elaborate-time quantization: Float -> FixedPoint -> I8E per row.
-- Safe for synthesis because inputs are structural constants.
quantizeMatI8E
  :: ( KnownNat cols)
  =>CArray2D rows cols                 -- Float params baked in the netlist
  -> QArray2D rows cols                 -- Float-free carrier for hardware
quantizeMatI8E (CArray2D rowsF) =
  let rowsFtoF = map (map realToFrac) rowsF
  in QArray2D { unQ2D = map quantizeI8E rowsFtoF }

-- Lightweight dequantization to FixedPoint for reuse of F-kernels.
dequantRowToF :: RowI8E n -> Vec n FixedPoint
dequantRowToF (mant, e) =
  let s = scalePow2F e 1
  in map (\q -> fromIntegral q * s) mant
