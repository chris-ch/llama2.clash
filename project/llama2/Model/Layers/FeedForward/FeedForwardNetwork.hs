module Model.Layers.FeedForward.FeedForwardNetwork (
   computeFeedForward
) where

import Clash.Prelude
import Model.Helpers.FixedPoint (rmsNormF)
import Model.Core.Types (CArray2D, ModelDimemsion, HiddenDimension)
import Model.Numeric.Types (FixedPoint)


import Clash.Prelude
import Model.Numeric.Types (FixedPoint)
import Model.Core.Types (ModelDimemsion)
import Model.Layers.Components.Quantized
    ( FeedForwardNetworkComponentQ(fRMSFfnF) )
import Model.Helpers.FixedPoint (rmsNormFwFix)
import Model.Layers.FeedForward.FeedForwardNetwork.Internal (runFeedForwardFQ)

computeFeedForward
  :: FeedForwardNetworkComponentQ
  -> Vec ModelDimemsion FixedPoint
  -> Vec ModelDimemsion FixedPoint
computeFeedForward ffn inputVector =
  let xHat    = rmsNormFwFix inputVector (fRMSFfnF ffn)
      ffnCore = runFeedForwardFQ ffn xHat
  in zipWith (+) inputVector ffnCore
