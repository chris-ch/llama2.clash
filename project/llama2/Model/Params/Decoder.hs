module Model.Params.Decoder (decoderConst) where

import Clash.Prelude
import Model.Config
  ( ModelDimension, HiddenDimension
  , HeadDimension
  , RotaryPositionalEmbeddingDimension, SequenceLength
  , VocabularySize
  )
import Model.Core.Types
  ( CArray2D(..)
  , SingleHeadComponent(..)
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..)
  )
import Model.Layers.TransformerLayer
  ( TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
  )
import Model.Layers.Components.Quantized
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
  { wqHead = zeroMatF @HeadDimension @ModelDimension
  , wkHead = zeroMatF @HeadDimension @ModelDimension
  , wvHead = zeroMatF @HeadDimension @ModelDimension
  , rotary = rotaryZerosF
  }

mhaFloat :: MultiHeadAttentionComponent
mhaFloat = MultiHeadAttentionComponent
  { heads  = repeat singleHeadF                 -- Vec NumQueryHeads
  , mWo    = repeat (zeroMatF @ModelDimension @HeadDimension)
  , rmsAtt = repeat 0.0                         -- Vec ModelDimension
  }

ffnFloat :: FeedForwardNetworkComponent
ffnFloat = FeedForwardNetworkComponent
  { fW1     = zeroMatF @HiddenDimension @ModelDimension
  , fW2     = zeroMatF @ModelDimension @HiddenDimension
  , fW3     = zeroMatF @HiddenDimension @ModelDimension
  , fRMSFfn = repeat 0.0
  }

embeddingFloat :: EmbeddingComponent
embeddingFloat = EmbeddingComponent
  { vocabulary     = zeroMatF @VocabularySize @ModelDimension
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
