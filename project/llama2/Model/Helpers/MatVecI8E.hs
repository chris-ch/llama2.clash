module Model.Helpers.MatVecI8E
  ( matrixVectorMultI8E_Fixed
  , dotRowI8E_Fixed
  ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D(..), RowI8E, dequantRowToF)
import Model.Helpers.Fixed (dotProductF)

-- Dot product: dequantize a row once, then reuse existing F dot-product.
dotRowI8E_Fixed :: KnownNat n => RowI8E n -> Vec n FixedPoint -> FixedPoint
dotRowI8E_Fixed row = dotProductF (dequantRowToF row)

-- Matrix @ vector where matrix is quantized (I8E rows) and vector is FixedPoint.
matrixVectorMultI8E_Fixed
  :: (KnownNat rows, KnownNat cols)
  => QArray2D rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMultI8E_Fixed (QArray2D rowsQ) xF =
  map (`dotRowI8E_Fixed` xF) rowsQ
