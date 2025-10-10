module LLaMa2.Core.Transformer.Internal (
  initialLayerData
) where

import Clash.Prelude
import LLaMa2.Core.Types ( LayerData(..) )
import LLaMa2.Config
  ( LLaMa2Dimension, NumQueryHeads, HeadDimension, NumKeyValueHeads
  )
import LLaMa2.Numeric.Types (FixedPoint)

initialLayerData :: LayerData
initialLayerData = LayerData
  { inputVector       = repeat 0          :: Vec LLaMa2Dimension FixedPoint
  , queryVectors      = repeat (repeat 0) :: Vec NumQueryHeads (Vec HeadDimension FixedPoint)
  , keyVectors        = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , valueVectors      = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , attentionOutput   = repeat 0          :: Vec LLaMa2Dimension FixedPoint
  , feedForwardOutput = repeat 0          :: Vec LLaMa2Dimension FixedPoint
  }
