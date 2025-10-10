module LLaMa2.Layers.Components.RotaryQ
  ( RotaryEncodingComponentF(..)
  , quantizeRotary
  ) where

import Clash.Prelude
import LLaMa2.Core.Types (CArray2D(..))
import LLaMa2.Config (SequenceLength, RotaryPositionalEmbeddingDimension)
import LLaMa2.Numeric.Types (FixedPoint)

data RotaryEncodingComponentF = RotaryEncodingComponentF
  { freqCosF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  , freqSinF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  } deriving (Generic, NFDataX, Show, Eq)

quantizeRotary :: (CArray2D SequenceLength RotaryPositionalEmbeddingDimension,
                   CArray2D SequenceLength RotaryPositionalEmbeddingDimension)
               -> RotaryEncodingComponentF
quantizeRotary (CArray2D cosF, CArray2D sinF) =
  RotaryEncodingComponentF
    { freqCosF = map (map realToFrac) cosF
    , freqSinF = map (map realToFrac) sinF
    }
