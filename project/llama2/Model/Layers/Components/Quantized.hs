module Model.Layers.Components.Quantized
  ( -- Types
    SingleHeadComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , FeedForwardNetworkComponentQ(..)
  , EmbeddingComponentQ(..)
  , MultiHeadAttentionComponent (..)
  , FeedForwardNetworkComponent (..)
    -- Converters (elaboration-time)
  , quantizeSingleHead
  , quantizeMHA
  , quantizeFFN
  , quantizeEmbedding
  ) where

import Clash.Prelude
import GHC.Generics (Generic)

import Model.Core.Types
  ( CArray2D(..)
  , ModelDimemsion
  , HiddenDimension
  , NumQueryHeads
  , HeadDimension
  , VocabularySize
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..), SingleHeadComponent (..)
  )
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D(..), quantizeMatI8E)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  , mWo    :: Vec NumQueryHeads (CArray2D ModelDimemsion HeadDimension)
  , rmsAtt :: Vec ModelDimemsion Float
  } deriving (Show)

data FeedForwardNetworkComponent = FeedForwardNetworkComponent
  { fW1 :: CArray2D HiddenDimension ModelDimemsion
  , fW2 :: CArray2D ModelDimemsion HiddenDimension
  , fW3 :: CArray2D HiddenDimension ModelDimemsion
  , fRMSFfn :: Vec ModelDimemsion Float
  } deriving (Show)

-- Float-free, quantized single head (per-row I8E weights).
data SingleHeadComponentQ = SingleHeadComponentQ
  { wqHeadQ :: QArray2D HeadDimension ModelDimemsion
  , wkHeadQ :: QArray2D HeadDimension ModelDimemsion
  , wvHeadQ :: QArray2D HeadDimension ModelDimemsion
  , rotaryQ :: RotaryEncodingComponent
  } deriving (Generic, Show, Eq)

-- MHA with quantized per-head WO and preconverted RMS weights.
data MultiHeadAttentionComponentQ = MultiHeadAttentionComponentQ
  { headsQ  :: Vec NumQueryHeads SingleHeadComponentQ
  , mWoQ    :: Vec NumQueryHeads (QArray2D ModelDimemsion HeadDimension)
  , rmsAttF :: Vec ModelDimemsion FixedPoint
  } deriving (Generic, Show, Eq)

-- FFN with quantized matrices and preconverted RMS.
data FeedForwardNetworkComponentQ = FeedForwardNetworkComponentQ
  { fW1Q     :: QArray2D HiddenDimension ModelDimemsion
  , fW2Q     :: QArray2D ModelDimemsion HiddenDimension
  , fW3Q     :: QArray2D HiddenDimension ModelDimemsion
  , fRMSFfnF :: Vec ModelDimemsion FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

-- Embedding with quantized vocabulary sized by the active VocabularySize alias.
data EmbeddingComponentQ = EmbeddingComponentQ
  { vocabularyQ     :: QArray2D VocabularySize ModelDimemsion
  , rmsFinalWeightF :: Vec ModelDimemsion FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

-- Elaborate-time converters (no Float in hardware).
quantizeSingleHead :: SingleHeadComponent -> SingleHeadComponentQ
quantizeSingleHead sh =
  SingleHeadComponentQ
    { wqHeadQ = quantizeMatI8E (wqHead sh)
    , wkHeadQ = quantizeMatI8E (wkHead sh)
    , wvHeadQ = quantizeMatI8E (wvHead sh)
    , rotaryQ = rotary sh
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

