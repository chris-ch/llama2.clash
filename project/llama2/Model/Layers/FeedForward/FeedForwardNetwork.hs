module Model.Layers.FeedForward.FeedForwardNetwork (
   computeFeedForward
) where

import Clash.Prelude
import Model.Helpers.FixedPoint ( rmsNormFwFix )
import Model.Core.Types ( CArray2D )
import Model.Config
    ( ModelDimension, HiddenDimension, ModelDimension )
import Model.Numeric.Types ( FixedPoint, FixedPoint )
import Model.Layers.Components.Quantized
    ( FeedForwardNetworkComponentQ(fRMSFfnF) )
import Model.Layers.FeedForward.FeedForwardNetwork.Internal (runFeedForward)

computeFeedForward
  :: FeedForwardNetworkComponentQ
  -> Vec ModelDimension FixedPoint
  -> Vec ModelDimension FixedPoint
computeFeedForward ffn inputVector =
  let xHat    = rmsNormFwFix inputVector (fRMSFfnF ffn)
      ffnCore = runFeedForward ffn xHat
  in zipWith (+) inputVector ffnCore
