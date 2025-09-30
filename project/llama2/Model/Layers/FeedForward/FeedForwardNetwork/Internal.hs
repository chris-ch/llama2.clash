module Model.Layers.FeedForward.FeedForwardNetwork.Internal  where

import Clash.Prelude
import Model.Core.Types (CArray2D, ModelDimemsion, HiddenDimension)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDimension ModelDimemsion
  , fW2 :: CArray2D ModelDimemsion HiddenDimension
  , fW3 :: CArray2D HiddenDimension ModelDimemsion
  , fRMSFfn :: Vec ModelDimemsion Float
  } deriving (Show)
