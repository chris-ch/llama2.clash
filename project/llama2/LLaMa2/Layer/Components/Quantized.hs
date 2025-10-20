module LLaMa2.Layer.Components.Quantized
  (
    -- Converters (elaboration-time)
  quantizeMHA
  , quantizeFFN
  , quantizeEmbedding
  ) where

import Clash.Prelude

import LLaMa2.Types.LayerData
  ( CArray2D(..)
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..), SingleHeadComponent (..), MultiHeadAttentionComponent (..), FeedForwardNetworkComponent (..)
  )
import LLaMa2.Types.ModelConfig  (
  RotaryPositionalEmbeddingDimension, SequenceLength
  )
import LLaMa2.Numeric.Quantization (quantizeMatI8E)
import LLaMa2.Types.Parameters

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
