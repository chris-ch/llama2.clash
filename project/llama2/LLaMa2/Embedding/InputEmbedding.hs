module LLaMa2.Embedding.InputEmbedding
 ( inputEmbedding
) where
import Clash.Prelude

import LLaMa2.Config
    ( ModelDimension, VocabularySize, ModelDimension, VocabularySize )
import LLaMa2.Core.Types ( Token, Token )
import LLaMa2.Numeric.Types ( FixedPoint, FixedPoint, scalePow2F )
import LLaMa2.Numeric.Quantization ( MatI8E, MatI8E )
import qualified LLaMa2.Layer.Components.Quantized as Quantized (EmbeddingComponentQ(..))

-- | Lookup token embedding from vocabulary
inputEmbedding
  :: HiddenClockResetEnable dom
  => Quantized.EmbeddingComponentQ           -- ^ Embedding parameters
  -> Signal dom Token                        -- ^ Input token
  -> Signal dom (Vec ModelDimension FixedPoint)  -- ^ Embedded vector
inputEmbedding embParams = embedder vocabulary
  where
    vocabulary :: MatI8E VocabularySize ModelDimension
    vocabulary = Quantized.vocabularyQ embParams

-- BRAM/ROM-backed dequantize-on-read. 1-cycle latency.
embedder
  :: HiddenClockResetEnable dom
  => MatI8E VocabularySize ModelDimension
  -> Signal dom Token
  -> Signal dom (Vec ModelDimension FixedPoint)
embedder table tokSig =
  let
    -- Precompute dequantized rows at elaboration time; stored in ROM.
    deqRow (mant, e) =
      let s = scalePow2F e 1
      in map (\q -> fromIntegral q * s) mant
    romContent :: Vec VocabularySize (Vec ModelDimension FixedPoint)
    romContent = map deqRow table
  in rom romContent tokSig
