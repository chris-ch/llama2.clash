module LLaMa2.Embedding.OutputProjection
 ( outputProjection
) where
import Clash.Prelude

import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Numeric.Operations (parallel32RowMatrixMultiplier)
import LLaMa2.Types.Parameters (DecoderParameters (..), EmbeddingComponentQ (..))
import LLaMa2.Types.ModelConfig  (ModelDimension, VocabularySize)
import LLaMa2.Numeric.Types (FixedPoint)

logitsProjector :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ validIn (readyPulse from pipeline)
  -> Signal dom Bool  -- ^ readyIn (always True for now, sampler is combinational)
  -> DecoderParameters
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec VocabularySize FixedPoint)  -- ^ logits output
     , Signal dom Bool  -- ^ validOut
     )
logitsProjector validIn readyIn decoder tokenVecSig =
  (logitsOut, validOut)
  where
    emb = modelEmbedding decoder

    -- Pre-normalize (combinational)
    tokenWithRms = rmsNormFwFix <$> tokenVecSig <*> pure (rmsFinalWeightF emb)

    -- Sequential matrix multiply (tied embeddings as classifier)
    (logitsOut, validOut, readyOut) =
      parallel32RowMatrixMultiplier validIn readyIn (vocabularyQ emb) tokenWithRms

-- | Simplified output projection wrapper
outputProjection
  :: HiddenClockResetEnable dom
  => DecoderParameters                           -- ^ Model parameters
  -> Signal dom Bool                             -- ^ layerComplete signal
  -> Signal dom (Vec ModelDimension FixedPoint)  -- ^ Final layer output
  -> ( Signal dom (Vec VocabularySize FixedPoint)  -- ^ Logits
     , Signal dom Bool                            -- ^ Logits valid
     )
outputProjection params layerComplete = logitsProjector
    layerComplete   -- validIn
    (pure True)     -- readyIn (always ready)
    params
