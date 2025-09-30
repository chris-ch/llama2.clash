module Model.Layers.FeedForward.FeedForwardNetwork.Internal  where

import Clash.Prelude

import Model.Core.Types (CArray2D, ModelDimemsion, HiddenDimension)
import Helpers (rmsNorm, matrixVectorMult)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDimension ModelDimemsion,
    fW2 :: CArray2D ModelDimemsion HiddenDimension,
    fW3 :: CArray2D HiddenDimension ModelDimemsion,
    fRMSFfn :: Vec ModelDimemsion Float
  } deriving (Show)

runFeedForward :: FeedForwardNetworkComponent -> Vec ModelDimemsion Float -> Vec ModelDimemsion Float
runFeedForward ffn xHat =
  let gate = map sigmoidLinearUnit $ matrixVectorMult (fW1 ffn) xHat
      up   = matrixVectorMult (fW3 ffn) xHat
  in matrixVectorMult (fW2 ffn) (zipWith (*) gate up)

-- Activation
sigmoidLinearUnit :: Float -> Float
sigmoidLinearUnit x = x / (1.0 + exp (-x))
