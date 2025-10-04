module Model.Helpers.MatVecI8E
  ( matrixVectorMult
  , sequentialMatVecStub
  ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D(..), RowI8E, dequantRowToF)
import Model.Helpers.FixedPoint (dotProductF)

-- Dot product: dequantize a row once, then reuse existing F dot-product.
dotRowI8E_Fixed :: KnownNat n => RowI8E n -> Vec n FixedPoint -> FixedPoint
dotRowI8E_Fixed row = dotProductF (dequantRowToF row)

-- Matrix @ vector where matrix is quantized (I8E rows) and vector is FixedPoint.
matrixVectorMult
  :: ( KnownNat cols)
  => QArray2D rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMult (QArray2D rowsQ) xF =
  map (`dotRowI8E_Fixed` xF) rowsQ

-- Ready/Valid sequential façade for matrix-vector multiplication
sequentialMatVecStub
  :: ( KnownNat cols )
  =>QArray2D rows cols
  -> Signal dom (Bool, Vec cols FixedPoint)      -- (validIn, inputVec)
  -> ( Signal dom (Vec rows FixedPoint)          -- outputVec
     , Signal dom Bool                           -- validOut
     , Signal dom Bool                           -- readyOut
     )
sequentialMatVecStub mat inSig =
  let
    (validIn, vecIn) = unbundle inSig
    res              = matrixVectorMult mat <$> vecIn
  in
    ( res
    , validIn        -- Stub: output valid matches input valid
    , pure True      -- Stub: always ready
    )
