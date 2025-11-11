module LLaMa2.Types.LayerData
  (
    LayerData (..),
    Token,
    Temperature,
    Seed,
    SingleHeadComponent (..),
    RotaryEncodingComponent (..),
    EmbeddingComponent (..),
    FeedForwardNetworkComponent(..),
    MultiHeadAttentionComponent(..),
    CArray2D (..),
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
    VocabularySize, HiddenDimension,
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

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show, Eq)

type Token = Unsigned 32

type Temperature = FixedPoint

type Seed = Unsigned 32

-- Data definitions for LLM architecture

data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: CArray2D VocabularySize ModelDimension,
    rmsFinalWeight :: Vec ModelDimension Float
  }
  deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: CArray2D SequenceLength RotaryPositionalEmbeddingDimension,
    freqSin :: CArray2D SequenceLength RotaryPositionalEmbeddingDimension
  }
  deriving (Show, Generic, Eq)

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: CArray2D HeadDimension ModelDimension,
    wkHead :: CArray2D HeadDimension ModelDimension,
    wvHead :: CArray2D HeadDimension ModelDimension,
    rotary :: RotaryEncodingComponent
  }
  deriving (Show)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  , mWo    :: Vec NumQueryHeads (CArray2D ModelDimension HeadDimension)
  , rmsAtt :: Vec ModelDimension Float
  } deriving (Show)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDimension ModelDimension
  , fW2 :: CArray2D ModelDimension HiddenDimension
  , fW3 :: CArray2D HiddenDimension ModelDimension
  , fRMSFfn :: Vec ModelDimension Float
  } deriving (Show)
