module LLaMa2.Layers.Components.Quantized
  ( -- Types
    SingleHeadComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , FeedForwardNetworkComponentQ(..)
  , EmbeddingComponentQ(..)
  , MultiHeadAttentionComponent (..)
  , FeedForwardNetworkComponent (..)
  , RotaryEncodingComponentF(..)
    -- Converters (elaboration-time)
  , quantizeMHA
  , quantizeFFN
  , quantizeEmbedding
  ) where

import Clash.Prelude

import LLaMa2.Core.Types
  ( CArray2D(..)
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..), SingleHeadComponent (..)
  )
import LLaMa2.Config (
  ModelDimension
  , HiddenDimension
  , NumQueryHeads
  , HeadDimension
  , VocabularySize, RotaryPositionalEmbeddingDimension, SequenceLength
  )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.ParamPack (MatI8E, quantizeMatI8E)

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

data RotaryEncodingComponentF = RotaryEncodingComponentF
  { freqCosF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  , freqSinF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  } deriving (Generic, NFDataX, Show, Eq)

-- Float-free, quantized single head (per-row I8E weights).
data SingleHeadComponentQ = SingleHeadComponentQ
  { wqHeadQ :: MatI8E HeadDimension ModelDimension
  , wkHeadQ :: MatI8E HeadDimension ModelDimension
  , wvHeadQ :: MatI8E HeadDimension ModelDimension
  , rotaryQ :: RotaryEncodingComponentF
  } deriving (Generic, Show, Eq)

-- MHA with quantized per-head WO and preconverted RMS weights.
data MultiHeadAttentionComponentQ = MultiHeadAttentionComponentQ
  { headsQ  :: Vec NumQueryHeads SingleHeadComponentQ
  , mWoQ    :: Vec NumQueryHeads (MatI8E ModelDimension HeadDimension)
  , rmsAttF :: Vec ModelDimension FixedPoint
  } deriving (Generic, Show, Eq)

-- FFN with quantized matrices and preconverted RMS.
data FeedForwardNetworkComponentQ = FeedForwardNetworkComponentQ
  { fW1Q     :: MatI8E HiddenDimension ModelDimension
  , fW2Q     :: MatI8E ModelDimension HiddenDimension
  , fW3Q     :: MatI8E HiddenDimension ModelDimension
  , fRMSFfnF :: Vec ModelDimension FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

-- Embedding with quantized vocabulary sized by the active VocabularySize alias.
data EmbeddingComponentQ = EmbeddingComponentQ
  { vocabularyQ     :: MatI8E VocabularySize ModelDimension
  , rmsFinalWeightF :: Vec ModelDimension FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

-- Elaborate-time converters (no Float in hardware).
quantizeSingleHead :: SingleHeadComponent -> SingleHeadComponentQ
quantizeSingleHead sh =
  SingleHeadComponentQ
    { wqHeadQ = quantizeMatI8E (wqHead sh)
    , wkHeadQ = quantizeMatI8E (wkHead sh)
    , wvHeadQ = quantizeMatI8E (wvHead sh)
    , rotaryQ = quantizeRotary (freqCos (rotary sh), freqSin (rotary sh))
    }

quantizeMHA :: MultiHeadAttentionComponent -> MultiHeadAttentionComponentQ
quantizeMHA mha =
  MultiHeadAttentionComponentQ
    { headsQ  = map quantizeSingleHead (heads mha)
    , mWoQ    = map quantizeMatI8E (mWo mha)
    , rmsAttF = map realToFrac (rmsAtt mha)
    }

quantizeFFN :: FeedForwardNetworkComponent -> FeedForwardNetworkComponentQ
quantizeFFN f =
  FeedForwardNetworkComponentQ
    { fW1Q     = quantizeMatI8E (fW1 f)
    , fW2Q     = quantizeMatI8E (fW2 f)
    , fW3Q     = quantizeMatI8E (fW3 f)
    , fRMSFfnF = map realToFrac (fRMSFfn f)
    }

quantizeEmbedding :: EmbeddingComponent -> EmbeddingComponentQ
quantizeEmbedding e =
  EmbeddingComponentQ
    { vocabularyQ     = quantizeMatI8E (vocabulary e)
    , rmsFinalWeightF = map realToFrac (rmsFinalWeight e)
    }

quantizeRotary :: (CArray2D SequenceLength RotaryPositionalEmbeddingDimension,
                   CArray2D SequenceLength RotaryPositionalEmbeddingDimension)
               -> RotaryEncodingComponentF
quantizeRotary (CArray2D cosF, CArray2D sinF) =
  RotaryEncodingComponentF
    { freqCosF = map (map realToFrac) cosF
    , freqSinF = map (map realToFrac) sinF
    }
