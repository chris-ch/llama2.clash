module Model.Core.Transformer.Internal where

import Clash.Prelude
import Helpers (liftA4)
import Model.Core.Types ( LayerData(..) )
import Model.Config
  ( NumLayers, VocabularySize, ModelDimension, SequenceLength, NumQueryHeads, HeadDimension, NumKeyValueHeads
  )
import Model.Numeric.Types (FixedPoint)

initialLayerData :: LayerData
initialLayerData = LayerData
  { inputVector       = repeat 0          :: Vec ModelDimension FixedPoint
  , queryVectors      = repeat (repeat 0) :: Vec NumQueryHeads (Vec HeadDimension FixedPoint)
  , keyVectors        = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , valueVectors      = repeat (repeat 0) :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , attentionOutput   = repeat 0          :: Vec ModelDimension FixedPoint
  , feedForwardOutput = repeat 0          :: Vec ModelDimension FixedPoint
  }
