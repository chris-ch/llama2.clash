module Model.Layers.FeedForward.FeedForwardNetwork (
   computeFeedForward
) where

import Clash.Prelude
import Model.Helpers.FixedPoint ( rmsNormFwFix )
import Model.Core.Types
    ( CArray2D, ModelDimemsion, HiddenDimension, ModelDimemsion )
import Model.Numeric.Types ( FixedPoint, FixedPoint )
import Model.Layers.Components.Quantized
    ( FeedForwardNetworkComponentQ(fRMSFfnF) )
import Model.Layers.FeedForward.FeedForwardNetwork.Internal (runFeedForwardFQ)

computeFeedForward
  :: FeedForwardNetworkComponentQ
  -> Vec ModelDimemsion FixedPoint
  -> Vec ModelDimemsion FixedPoint
computeFeedForward ffn inputVector =
  let xHat    = rmsNormFwFix inputVector (fRMSFfnF ffn)
      ffnCore = runFeedForwardFQ ffn xHat
  in zipWith (+) inputVector ffnCore
