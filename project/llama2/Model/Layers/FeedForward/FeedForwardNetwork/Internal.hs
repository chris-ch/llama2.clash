module Model.Layers.FeedForward.FeedForwardNetwork.Internal  where

import Clash.Prelude
import Model.Core.Types
    ( CArray2D,
      ModelDimemsion,
      HiddenDimension,
      ModelDimemsion,
      HiddenDimension )

import Model.Numeric.Types (FixedPoint)
import Model.Helpers.MatVecI8E (matrixVectorMultI8E_Fixed)
import Model.Helpers.FixedPoint (rmsNormF)
import qualified Model.Numeric.Fixed
import Model.Layers.Components.Quantized (FeedForwardNetworkComponentQ (..))

-- Same topology as before, but weights are I8E and mat-vec is quantized.
runFeedForwardFQ
  :: FeedForwardNetworkComponentQ
  -> Vec ModelDimemsion FixedPoint
  -> Vec ModelDimemsion FixedPoint
runFeedForwardFQ ffn xHat =
  let gate = map sigmoidLinearUnitF $ matrixVectorMultI8E_Fixed (fW1Q ffn) xHat
      up   =                           matrixVectorMultI8E_Fixed (fW3Q ffn) xHat
  in  matrixVectorMultI8E_Fixed (fW2Q ffn) (zipWith (*) gate up)

sigmoidLinearUnitF :: FixedPoint -> FixedPoint
sigmoidLinearUnitF x = x / (1 + expF (negate x))
  where
    -- reuse your expF definition
    expF = Model.Numeric.Fixed.expF
