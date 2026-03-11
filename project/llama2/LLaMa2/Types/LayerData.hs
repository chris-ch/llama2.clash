module LLaMa2.Types.LayerData
  (
    LayerData (..),
    Token,
    Temperature,
    Seed
  )
where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( HeadDimension,
    ModelDimension,
    NumKeyValueHeads,
    NumQueryHeads,
  )
import LLaMa2.Numeric.Types (FixedPoint)

-- ============================================================================
-- Intermediate Data Storage
-- ============================================================================

-- Per-layer intermediate data vectors carried through the pipeline.
-- Updated selectively depending on cycle stage.
data LayerData = LayerData
  { inputVector :: Vec ModelDimension FixedPoint,
    queryVectors :: Vec NumQueryHeads (Vec HeadDimension FixedPoint),
    keyVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint),
    valueVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint),
    attentionOutput :: Vec ModelDimension FixedPoint,
    feedForwardOutput :: Vec ModelDimension FixedPoint
  }
  deriving (Show, Generic, NFDataX, Eq)

type Token = Unsigned 32

type Temperature = FixedPoint

type Seed = Unsigned 32
