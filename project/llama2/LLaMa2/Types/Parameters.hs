module LLaMa2.Types.Parameters (
    DecoderParameters(..)
) where
import Clash.Explicit.Prelude

import LLaMa2.Layer.Components.Quantized (EmbeddingComponentQ)
import LLaMa2.Config (NumLayers)
import LLaMa2.Layer.TransformerLayer (TransformerLayerComponent)

data DecoderParameters = DecoderParameters
  { modelEmbedding :: EmbeddingComponentQ
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)
