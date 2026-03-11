module Simulation.SimulationTypes (
    EmbeddingComponent(..),
    SingleHeadComponent (..),
    MultiHeadAttentionComponent(..),
    FeedForwardNetworkComponent(..),
    RotaryEncodingComponent (..),
    CArray2D (..),
    ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (VocabularySize, ModelDimension, HeadDimension, NumQueryHeads, SequenceLength, RotaryPositionalEmbeddingDimension, HiddenDimension)

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

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show, Eq)
