module Model.Layers.FeedForward.FeedForwardNetwork (
    FeedForwardNetworkComponent(..), computeFeedForward
) where

import Clash.Prelude
import Model.Layers.FeedForward.FeedForwardNetwork.Internal
import Model.Helpers.Fixed (rmsNormF)
import Model.Core.Types (CArray2D, ModelDimemsion, HiddenDimension)
import Model.Numeric.Types (FixedPoint)

computeFeedForward :: FeedForwardNetworkComponent
  -> Vec ModelDimemsion FixedPoint
  -> Vec ModelDimemsion FixedPoint
computeFeedForward ffn inputVector =
  let
    xHat     = rmsNormF inputVector (fRMSFfn ffn)
    ffnCore  = runFeedForwardF ffn xHat
  in zipWith (+) inputVector ffnCore
