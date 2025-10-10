module LLaMa2.Params.Decoder (decoderConst) where

import Clash.Prelude
import LLaMa2.Config
  ( LLaMa2Dimension, HiddenDimension
  , HeadDimension
  , RotaryPositionalEmbeddingDimension, SequenceLength
  , VocabularySize
  )
import LLaMa2.Core.Types
  ( CArray2D(..)
  , SingleHeadComponent(..)
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..)
  )
import LLaMa2.Layers.TransformerLayer
  ( TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
  )
import LLaMa2.Layers.Components.Quantized
  ( MultiHeadAttentionComponent(..)
  , FeedForwardNetworkComponent(..)
  , quantizeMHA, quantizeFFN, quantizeEmbedding
  )

-- Helpers: zero-filled Float matrices/vectors to make the design elaborate now.
zeroMatF :: forall rows cols. (KnownNat rows, KnownNat cols) => CArray2D rows cols
zeroMatF = CArray2D (repeat (repeat 0.0))

rotaryZerosF :: RotaryEncodingComponent
rotaryZerosF = RotaryEncodingComponent
  { freqCos = zeroMatF @SequenceLength @RotaryPositionalEmbeddingDimension
  , freqSin = zeroMatF @SequenceLength @RotaryPositionalEmbeddingDimension
  }

singleHeadF :: SingleHeadComponent
singleHeadF = SingleHeadComponent
  { wqHead = zeroMatF @HeadDimension @LLaMa2Dimension
  , wkHead = zeroMatF @HeadDimension @LLaMa2Dimension
  , wvHead = zeroMatF @HeadDimension @LLaMa2Dimension
  , rotary = rotaryZerosF
  }

mhaFloat :: MultiHeadAttentionComponent
mhaFloat = MultiHeadAttentionComponent
  { heads  = repeat singleHeadF                 -- Vec NumQueryHeads
  , mWo    = repeat (zeroMatF @LLaMa2Dimension @HeadDimension)
  , rmsAtt = repeat 0.0                         -- Vec LLaMa2Dimension
  }

ffnFloat :: FeedForwardNetworkComponent
ffnFloat = FeedForwardNetworkComponent
  { fW1     = zeroMatF @HiddenDimension @LLaMa2Dimension
  , fW2     = zeroMatF @LLaMa2Dimension @HiddenDimension
  , fW3     = zeroMatF @HiddenDimension @LLaMa2Dimension
  , fRMSFfn = repeat 0.0
  }

embeddingFloat :: EmbeddingComponent
embeddingFloat = EmbeddingComponent
  { vocabulary     = zeroMatF @VocabularySize @LLaMa2Dimension
  , rmsFinalWeight = repeat 0.0
  }

-- Public constant: quantized components embedded in hardware.
decoderConst :: TransformerDecoderComponent
decoderConst =
  let layerQ = TransformerLayerComponent
                { multiHeadAttention = quantizeMHA mhaFloat
                , feedforwardNetwork = quantizeFFN ffnFloat
                }
  in TransformerDecoderComponent
        { modelEmbedding = quantizeEmbedding embeddingFloat
        , modelLayers    = repeat layerQ  -- Vec NumLayers
        }
