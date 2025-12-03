module Simulation.Parameters (
    DecoderParameters(..), TransformerLayerComponent(..)
  , MultiHeadAttentionComponentQ(..)
  , FeedForwardNetworkComponentQ(..)
  , EmbeddingComponentQ(..)
  , RotaryEncodingComponentF(..)
  , KeyValueHeadComponentQ(..)
  , QueryHeadComponentQ(..)
  , quantizeMHA, quantizeFFN, quantizeEmbedding, quantizeRotary
) where
import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, NumQueryHeads, ModelDimension, HeadDimension, HiddenDimension, SequenceLength, RotaryPositionalEmbeddingDimension, VocabularySize, NumKeyValueHeads)
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.LayerData (FeedForwardNetworkComponent (..), EmbeddingComponent (..), CArray2D (..), MultiHeadAttentionComponent (..), SingleHeadComponent (..))
import Simulation.ParametersQuantization (quantizeMatI8E)

data DecoderParameters = DecoderParameters
  { modelEmbedding :: EmbeddingComponentQ
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  , rotaryEncoding :: RotaryEncodingComponentF
  } deriving (Show)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttentionComponentQ
  , feedforwardNetwork :: FeedForwardNetworkComponentQ
  } deriving (Show)

data RotaryEncodingComponentF = RotaryEncodingComponentF
  { freqCosF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  , freqSinF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  } deriving (Generic, NFDataX, Show, Eq)

-- MHA with quantized per-head WO and preconverted RMS weights.
-- Query head only contains Q matrix and rotary
newtype QueryHeadComponentQ
  = QueryHeadComponentQ {qMatrix :: MatI8E HeadDimension ModelDimension}
  deriving (Generic, Show, Eq)

-- KV head contains shared K and V matrices
data KeyValueHeadComponentQ = KeyValueHeadComponentQ
  { kMatrix :: MatI8E HeadDimension ModelDimension
  , vMatrix :: MatI8E HeadDimension ModelDimension
  } deriving (Generic, Show, Eq)

-- MHA structure with explicit separation
data MultiHeadAttentionComponentQ = MultiHeadAttentionComponentQ
  { qHeads  :: Vec NumQueryHeads QueryHeadComponentQ        -- 8 Q heads
  , kvHeads :: Vec NumKeyValueHeads KeyValueHeadComponentQ  -- 4 KV heads
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
quantizeMHA :: MultiHeadAttentionComponent -> MultiHeadAttentionComponentQ
quantizeMHA mha =
  let allHeads = heads mha
      -- Convert type-level nats to runtime Int
      numQHeads = natToNum @NumQueryHeads :: Int
      numKVHeads = natToNum @NumKeyValueHeads :: Int
      groupSize = numQHeads `div` numKVHeads

      -- Extract unique KV heads (every Nth head where N = groupSize)
      kvHeadIndices :: Vec NumKeyValueHeads Int
      kvHeadIndices = map (* groupSize) (iterateI (+1) 0)

  in MultiHeadAttentionComponentQ
    { qHeads  = map (QueryHeadComponentQ . quantizeMatI8E . wqHead)
                    allHeads
    , kvHeads = map (\i -> KeyValueHeadComponentQ
                      (quantizeMatI8E (wkHead (allHeads !! i)))
                      (quantizeMatI8E (wvHead (allHeads !! i))))
                    kvHeadIndices
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
