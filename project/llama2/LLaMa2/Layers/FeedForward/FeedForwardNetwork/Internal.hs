module LLaMa2.Layers.FeedForward.FeedForwardNetwork.Internal  (
  runFeedForward
  , sigmoidLinearUnitF
)where

import Clash.Prelude
import LLaMa2.Config
    (
      LLaMa2Dimension,
      LLaMa2Dimension )

import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Helpers.MatVecI8E (matrixVectorMult)
import qualified LLaMa2.Numeric.Fixed
import LLaMa2.Layers.Components.Quantized (FeedForwardNetworkComponentQ (..))

-- Same topology as before, but weights are I8E and mat-vec is quantized.
runFeedForward
  :: FeedForwardNetworkComponentQ
  -> Vec LLaMa2Dimension FixedPoint
  -> Vec LLaMa2Dimension FixedPoint
runFeedForward ffn xHat =
  let gate = map sigmoidLinearUnitF $ matrixVectorMult (fW1Q ffn) xHat
      up   =                           matrixVectorMult (fW3Q ffn) xHat
  in  matrixVectorMult (fW2Q ffn) (zipWith (*) gate up)

sigmoidLinearUnitF :: FixedPoint -> FixedPoint
sigmoidLinearUnitF x = x / (1 + expF (negate x))
  where
    -- reuse your expF definition
    expF = LLaMa2.Numeric.Fixed.expF
