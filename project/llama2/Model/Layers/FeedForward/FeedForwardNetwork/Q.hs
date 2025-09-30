module Model.Layers.FeedForward.FeedForwardNetwork.Q
  ( computeFeedForwardQ ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint)
import Model.Core.Types (ModelDimemsion)
import Model.Layers.FeedForward.FeedForwardNetwork.InternalQ
import Model.Layers.Components.Quantized
import Model.Helpers.FixedPoint (rmsNormFwFix)

computeFeedForwardQ
  :: FeedForwardNetworkComponentQ
  -> Vec ModelDimemsion FixedPoint
  -> Vec ModelDimemsion FixedPoint
computeFeedForwardQ ffn inputVector =
  let xHat    = rmsNormFwFix inputVector (fRMSFfnF ffn)
      ffnCore = runFeedForwardFQ ffn xHat
  in zipWith (+) inputVector ffnCore
