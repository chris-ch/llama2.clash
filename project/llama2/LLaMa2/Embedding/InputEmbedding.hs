module LLaMa2.Embedding.InputEmbedding
 ( inputEmbedding
) where
import Clash.Prelude

import LLaMa2.Config (ModelDimension, VocabularySize)
import LLaMa2.Core.Types (Token)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.ParamPack (MatI8E)
import qualified LLaMa2.Layer.Components.Quantized as Quantized (EmbeddingComponentQ(..))
import qualified LLaMa2.Core.Embedding as Embedding (embedder)

-- | Lookup token embedding from vocabulary
inputEmbedding
  :: HiddenClockResetEnable dom
  => Quantized.EmbeddingComponentQ           -- ^ Embedding parameters
  -> Signal dom Token                        -- ^ Input token
  -> Signal dom (Vec ModelDimension FixedPoint)  -- ^ Embedded vector
inputEmbedding embParams = Embedding.embedder vocabulary
  where
    vocabulary :: MatI8E VocabularySize ModelDimension
    vocabulary = Quantized.vocabularyQ embParams
