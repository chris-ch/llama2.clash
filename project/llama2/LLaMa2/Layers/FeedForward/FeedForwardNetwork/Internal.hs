module LLaMa2.Layers.FeedForward.FeedForwardNetwork.Internal  (
  feedForwardCore
  , sigmoidLinearUnitF
)where

import Clash.Prelude
import LLaMa2.Config
    (
      ModelDimension,
      ModelDimension )

import LLaMa2.Numeric.Types (FixedPoint)
import Simulation.MatVecSim (matrixVectorMult)
import qualified LLaMa2.Numeric.Fixed
import LLaMa2.Layers.Components.Quantized (FeedForwardNetworkComponentQ (..))

-- Same topology as before, but weights are I8E and mat-vec is quantized.
feedForwardCore :: FeedForwardNetworkComponentQ
  -> Vec ModelDimension FixedPoint
  -> Vec ModelDimension FixedPoint
feedForwardCore ffn xHat =
  let gate = map sigmoidLinearUnitF $ matrixVectorMult (fW1Q ffn) xHat
      up   =                           matrixVectorMult (fW3Q ffn) xHat
  in  matrixVectorMult (fW2Q ffn) (zipWith (*) gate up)

sigmoidLinearUnitF :: FixedPoint -> FixedPoint
sigmoidLinearUnitF x = x / (1 + expF (negate x))
  where
    -- reuse your expF definition
    expF = LLaMa2.Numeric.Fixed.expF
