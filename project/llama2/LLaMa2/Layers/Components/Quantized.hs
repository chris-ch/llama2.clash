module LLaMa2.Layers.Components.Quantized
  ( -- Types
    SingleHeadComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , FeedForwardNetworkComponentQ(..)
  , EmbeddingComponentQ(..)
  , MultiHeadAttentionComponent (..)
  , FeedForwardNetworkComponent (..)
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
  LLaMa2Dimension
  , HiddenDimension
  , NumQueryHeads
  , HeadDimension
  , VocabularySize
  )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.ParamPack (MatI8E, quantizeMatI8E)
import LLaMa2.Layers.Components.RotaryQ (quantizeRotary, RotaryEncodingComponentF)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  , mWo    :: Vec NumQueryHeads (CArray2D LLaMa2Dimension HeadDimension)
  , rmsAtt :: Vec LLaMa2Dimension Float
  } deriving (Show)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDimension LLaMa2Dimension
  , fW2 :: CArray2D LLaMa2Dimension HiddenDimension
  , fW3 :: CArray2D HiddenDimension LLaMa2Dimension
  , fRMSFfn :: Vec LLaMa2Dimension Float
  } deriving (Show)

-- Float-free, quantized single head (per-row I8E weights).
data SingleHeadComponentQ = SingleHeadComponentQ
  { wqHeadQ :: MatI8E HeadDimension LLaMa2Dimension
  , wkHeadQ :: MatI8E HeadDimension LLaMa2Dimension
  , wvHeadQ :: MatI8E HeadDimension LLaMa2Dimension
  , rotaryQ :: RotaryEncodingComponentF
  } deriving (Generic, Show, Eq)

-- MHA with quantized per-head WO and preconverted RMS weights.
data MultiHeadAttentionComponentQ = MultiHeadAttentionComponentQ
  { headsQ  :: Vec NumQueryHeads SingleHeadComponentQ
  , mWoQ    :: Vec NumQueryHeads (MatI8E LLaMa2Dimension HeadDimension)
  , rmsAttF :: Vec LLaMa2Dimension FixedPoint
  } deriving (Generic, Show, Eq)

-- FFN with quantized matrices and preconverted RMS.
data FeedForwardNetworkComponentQ = FeedForwardNetworkComponentQ
  { fW1Q     :: MatI8E HiddenDimension LLaMa2Dimension
  , fW2Q     :: MatI8E LLaMa2Dimension HiddenDimension
  , fW3Q     :: MatI8E HiddenDimension LLaMa2Dimension
  , fRMSFfnF :: Vec LLaMa2Dimension FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

-- Embedding with quantized vocabulary sized by the active VocabularySize alias.
data EmbeddingComponentQ = EmbeddingComponentQ
  { vocabularyQ     :: MatI8E VocabularySize LLaMa2Dimension
  , rmsFinalWeightF :: Vec LLaMa2Dimension FixedPoint
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

