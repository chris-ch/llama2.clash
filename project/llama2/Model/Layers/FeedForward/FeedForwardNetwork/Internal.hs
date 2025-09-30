module Model.Layers.FeedForward.FeedForwardNetwork.Internal  where

import Clash.Prelude
import Model.Helpers.FixedPoint (matrixVectorMultF, rmsNormF)
import Model.Core.Types (CArray2D, ModelDimemsion, HiddenDimension)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.Fixed (expF)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDimension ModelDimemsion
  , fW2 :: CArray2D ModelDimemsion HiddenDimension
  , fW3 :: CArray2D HiddenDimension ModelDimemsion
  , fRMSFfn :: Vec ModelDimemsion Float
  } deriving (Show)
  
runFeedForwardF :: FeedForwardNetworkComponent -> Vec ModelDimemsion FixedPoint -> Vec ModelDimemsion FixedPoint
runFeedForwardF ffn xHat =
  let gate = map sigmoidLinearUnitF $ matrixVectorMultF (fW1 ffn) xHat
      up   = matrixVectorMultF (fW3 ffn) xHat
  in  matrixVectorMultF (fW2 ffn) (zipWith (*) gate up)

sigmoidLinearUnitF :: FixedPoint -> FixedPoint
sigmoidLinearUnitF x = x / (1 + expF (negate x))
