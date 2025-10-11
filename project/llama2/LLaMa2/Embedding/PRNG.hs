module LLaMa2.Embedding.PRNG (
    tokenSampler
) where

import Clash.Prelude
import Data.Maybe (fromMaybe)
import LLaMa2.Core.Types
  ( Temperature, LayerData (..), Token, Seed)
import LLaMa2.Config ( VocabularySize , ModelDimension )

import qualified LLaMa2.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent(..))
import qualified Clash.Sized.Vector as CV
import LLaMa2.Helpers.FixedPoint (rmsNormFwFix)
import Simulation.MatVecSim (matrixVectorMult)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Fixed (expF)
import LLaMa2.Layers.Components.Quantized (EmbeddingComponentQ(..))

-- xorshift32 unchanged
xorshift32 :: Unsigned 32 -> Unsigned 32
xorshift32 s0 =
  let s1 = s0 `xor` shiftL s0 13
      s2 = s1 `xor` shiftR s1 17
      s3 = s2 `xor` shiftL s2 5
  in s3

-- classifier logits in FixedPoint using quantized tied embeddings
transformerLogits :: TransformerLayer.TransformerDecoderComponent
                   -> Vec ModelDimension FixedPoint
                   -> Vec VocabularySize FixedPoint
transformerLogits decoder tokenVector =
  let emb = TransformerLayer.modelEmbedding decoder            -- EmbeddingComponentQ
      tokenWithRms = rmsNormFwFix tokenVector (rmsFinalWeightF emb)
  in matrixVectorMult (vocabularyQ emb) tokenWithRms

logitsConverter :: TransformerLayer.TransformerDecoderComponent
              -> Signal dom LayerData
              -> Signal dom (Vec VocabularySize FixedPoint)
logitsConverter decoder nextLayerDataSignal =
  transformerLogits decoder . feedForwardOutput <$> nextLayerDataSignal

readyPulseRegister :: forall dom. HiddenClockResetEnable dom
   => Signal dom Bool -> Signal dom Bool
readyPulseRegister readyPulseSignal = regEn True readyPulseSignal (pure False)

seedMixer :: Signal dom (Unsigned 32) -> Signal dom (Unsigned 32)
seedMixer seedSignal = (`xor` 0x9E3779B9) <$> seedSignal

pseudoRandomGenerator :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom (Unsigned 32)
pseudoRandomGenerator readyPulse seedSignal =
  let nextVal = mux (readyPulseRegister readyPulse) (xorshift32 <$> seedMixer seedSignal)
                                      (xorshift32 <$> pseudoRandomGenerator readyPulse seedSignal)
  in regEn 2463534242 readyPulse nextVal

uniformRandom01Generator :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom FixedPoint
uniformRandom01Generator readyPulse seed =
  (/ realToFrac (16777216.0 :: Double)) . fromIntegral . (`shiftR` 8)
    <$> pseudoRandomGenerator readyPulse seed

-- Top-level sampler in FixedPoint
tokenSampler :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> TransformerLayer.TransformerDecoderComponent
  -> Signal dom LayerData
  -> Signal dom Token
tokenSampler readyPulse temperature seed decoder nextLayerData =
  liftA3
    (\temperature' logits u ->
        if temperature' <= 0 then argMax logits
        else let probabilities = softmax temperature' logits
             in sampleFromProbs u probabilities)
    temperature (logitsConverter decoder nextLayerData) (uniformRandom01Generator readyPulse seed)

-- categorical sampling from FixedPoint probabilities
sampleFromProbs :: forall n. (KnownNat (n + 1), KnownNat n) => FixedPoint -> Vec (n + 1) FixedPoint -> Unsigned 32
sampleFromProbs u probs =
  let cdf = CV.scanl1 (+) probs
      idxM = findIndex (>= u) cdf
  in fromIntegral (fromEnum (fromMaybe maxBound idxM))

-- Fixed softmax
softmax :: forall n. KnownNat (n + 1) => FixedPoint -> Vec (n + 1) FixedPoint -> Vec (n + 1) FixedPoint
softmax t xs =
  let m    = maximum xs
      exps = map (\x -> expF ((x - m) / t)) xs
      s    = sum exps
  in map (/ s) exps

-- argmax unchanged (returns Token)
argMax :: forall n. (KnownNat (n + 1)) => Vec (n+1) FixedPoint -> Unsigned 32
argMax vec =
  let (ix, _) = foldl (\(iBest, vBest) (i, v) -> if v > vBest then (i, v) else (iBest, vBest))
                      (0, head vec)
                      (imap (\i x -> (fromIntegral i, x)) vec)
  in ix
