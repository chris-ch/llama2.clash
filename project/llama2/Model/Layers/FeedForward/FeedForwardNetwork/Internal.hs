module Model.Layers.FeedForward.FeedForwardNetwork.Internal  (
  runFeedForward
  , sigmoidLinearUnitF
)where

import Clash.Prelude
import Model.Config
    (
      ModelDimension,
      ModelDimension )

import Model.Numeric.Types (FixedPoint)
import Model.Helpers.MatVecI8E (matrixVectorMult)
import qualified Model.Numeric.Fixed
import Model.Layers.Components.Quantized (FeedForwardNetworkComponentQ (..))

-- Same topology as before, but weights are I8E and mat-vec is quantized.
runFeedForward
  :: FeedForwardNetworkComponentQ
  -> Vec ModelDimension FixedPoint
  -> Vec ModelDimension FixedPoint
runFeedForward ffn xHat =
  let gate = map sigmoidLinearUnitF $ matrixVectorMult (fW1Q ffn) xHat
      up   =                           matrixVectorMult (fW3Q ffn) xHat
  in  matrixVectorMult (fW2Q ffn) (zipWith (*) gate up)

sigmoidLinearUnitF :: FixedPoint -> FixedPoint
sigmoidLinearUnitF x = x / (1 + expF (negate x))
  where
    -- reuse your expF definition
    expF = Model.Numeric.Fixed.expF
