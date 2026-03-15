module LLaMa2.Types.LayerData
  (
    LayerData (..),
    Token,
    Temperature,
    Seed,
    RotaryEncodingComponentF (..)
  )
where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( HeadDimension,
    ModelDimension,
    NumKeyValueHeads,
    NumQueryHeads,
    RotaryPositionalEmbeddingDimension,
    SequenceLength,
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
  deriving (Show, Generic, Eq, NFDataX, BitPack)

type Token = Unsigned 32

type Seed = Unsigned 32

type Temperature = FixedPoint

-- Precomputed rotary positional encoding tables (hardware-compatible type).
data RotaryEncodingComponentF = RotaryEncodingComponentF
  { freqCosF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  , freqSinF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  } deriving (Generic, NFDataX, Show, Eq)
