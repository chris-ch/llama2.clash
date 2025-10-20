module LLaMa2.Types.Parameters (
    DecoderParameters(..)
) where
import Clash.Prelude

import LLaMa2.Layer.Components.Quantized (EmbeddingComponentQ)
import LLaMa2.Types.ModelConfig (NumLayers)
import LLaMa2.Layer.TransformerLayer (TransformerLayerComponent)

data DecoderParameters = DecoderParameters
  { modelEmbedding :: EmbeddingComponentQ
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)
