module Model.Layers.FeedForward.FeedForwardNetwork (
    FeedForwardNetworkComponent(..)
) where

import Clash.Prelude
import Model.Layers.FeedForward.FeedForwardNetwork.Internal
    ( FeedForwardNetworkComponent(..) )
import Model.Helpers.FixedPoint (rmsNormF)
import Model.Core.Types (CArray2D, ModelDimemsion, HiddenDimension)
import Model.Numeric.Types (FixedPoint)

