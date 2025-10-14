module LLaMa2.Layers.FeedForward.FeedForwardNetwork (
   feedForwardStage
) where

import Clash.Prelude
import LLaMa2.Helpers.FixedPoint ( rmsNormFwFix )
import LLaMa2.Config
    ( ModelDimension, ModelDimension )
import LLaMa2.Numeric.Types ( FixedPoint, FixedPoint )
import LLaMa2.Layers.Components.Quantized
    ( FeedForwardNetworkComponentQ(fRMSFfnF) )
import LLaMa2.Layers.FeedForward.FeedForwardNetwork.Internal (feedForwardCore)

feedForwardStage
  :: FeedForwardNetworkComponentQ
  -> Vec ModelDimension FixedPoint
  -> Vec ModelDimension FixedPoint
feedForwardStage ffn inputVector =
  let xHat    = rmsNormFwFix inputVector (fRMSFfnF ffn)
      ffnCore = feedForwardCore ffn xHat
  in zipWith (+) inputVector ffnCore
