module LLaMa2.Layer.FeedForward.Activation (
   sigmoidLinearUnit
) where

import Clash.Prelude

import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Numeric.Quantization

sigmoidLinearUnit :: FixedPoint -> FixedPoint
sigmoidLinearUnit x = x / (1 + expF (negate x))
  where
    -- reuse your expF definition
    expF = LLaMa2.Numeric.Quantization.expF
