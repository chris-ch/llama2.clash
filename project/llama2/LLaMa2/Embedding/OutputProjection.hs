module LLaMa2.Embedding.OutputProjection
 ( logitsProjector
) where
import Clash.Prelude

import LLaMa2.Helpers.FixedPoint (rmsNormFwFix)
import LLaMa2.Layer.Components.Quantized (EmbeddingComponentQ(..))
import LLaMa2.Helpers.MatVecI8E (parallel32RowMatrixMultiplier)
import LLaMa2.Types.Parameters (DecoderParameters (..))
import LLaMa2.Config (ModelDimension, VocabularySize)
import LLaMa2.Numeric.Types (FixedPoint)

logitsProjector :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ validIn (readyPulse from pipeline)
  -> Signal dom Bool  -- ^ readyIn (always True for now, sampler is combinational)
  -> DecoderParameters
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec VocabularySize FixedPoint)  -- ^ logits output
     , Signal dom Bool  -- ^ validOut
     , Signal dom Bool  -- ^ readyOut
     )
logitsProjector validIn readyIn decoder tokenVecSig =
  (logitsOut, validOut, readyOut)
  where
    emb = modelEmbedding decoder

    -- Pre-normalize (combinational)
    tokenWithRms = rmsNormFwFix <$> tokenVecSig <*> pure (rmsFinalWeightF emb)

    -- Sequential matrix multiply (tied embeddings as classifier)
    (logitsOut, validOut, readyOut) =
      parallel32RowMatrixMultiplier validIn readyIn (vocabularyQ emb) tokenWithRms
