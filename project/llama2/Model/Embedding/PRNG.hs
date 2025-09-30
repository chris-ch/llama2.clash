module Model.Embedding.PRNG (
    sampledTokenSignal
) where

import Clash.Prelude
import Data.Maybe (fromMaybe)
import Model.Core.Types
  ( Temperature, IntermediateData (..), Token, VocabularySize
  , CArray2D (..), ModelDimemsion, EmbeddingComponent (..))
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent(..))
import qualified Clash.Sized.Vector as CV
import Model.Helpers.Fixed (rmsNormF, dotProductF)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.Fixed (expF)

-- xorshift32 unchanged
xorshift32 :: Unsigned 32 -> Unsigned 32
xorshift32 s0 =
  let s1 = s0 `xor` shiftL s0 13
      s2 = s1 `xor` shiftR s1 17
      s3 = s2 `xor` shiftL s2 5
  in s3

-- classifier logits in FixedPoint
transformerLogitsF :: TransformerLayer.TransformerDecoderComponent
                   -> Vec ModelDimemsion FixedPoint
                   -> Vec VocabularySize FixedPoint
transformerLogitsF decoder tokenVector =
  let vocab = vocabulary (TransformerLayer.modelEmbedding decoder)
      rmsWeight = rmsFinalWeight (TransformerLayer.modelEmbedding decoder)
      tokenWithRms = rmsNormF tokenVector rmsWeight
      CArray2D vocabRows = vocab
      vocabRowsF = map (map realToFrac) vocabRows :: Vec VocabularySize (Vec ModelDimemsion FixedPoint)
  in map (dotProductF tokenWithRms) vocabRowsF

logitsSignalF :: TransformerLayer.TransformerDecoderComponent
              -> Signal dom IntermediateData
              -> Signal dom (Vec VocabularySize FixedPoint)
logitsSignalF decoder nextIntermediateDataSignal =
  transformerLogitsF decoder . feedForwardOutput <$> nextIntermediateDataSignal

-- PRNG state (unchanged API, now returns FixedPoint uniform)
firstPulseSignal :: forall dom. HiddenClockResetEnable dom
   => Signal dom Bool -> Signal dom Bool
firstPulseSignal readyPulseSignal = regEn True readyPulseSignal (pure False)

mixedSeedSignal :: Signal dom (Unsigned 32) -> Signal dom (Unsigned 32)
mixedSeedSignal seedSignal = (`xor` 0x9E3779B9) <$> seedSignal

prngStateSignal :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom (Unsigned 32)
prngStateSignal readyPulseSignal seedSignal =
  let nextVal = mux (firstPulseSignal readyPulseSignal) (xorshift32 <$> mixedSeedSignal seedSignal)
                                      (xorshift32 <$> prngStateSignal readyPulseSignal seedSignal)
  in regEn 2463534242 readyPulseSignal nextVal

uniformRandom01SignalF :: forall dom. HiddenClockResetEnable dom
  => Signal dom Bool -> Signal dom (Unsigned 32) -> Signal dom FixedPoint
uniformRandom01SignalF readyPulseSignal seedSignal =
  (/ realToFrac (16777216.0 :: Double)) . fromIntegral . (`shiftR` 8)
    <$> prngStateSignal readyPulseSignal seedSignal

-- Fixed softmax
softmaxF :: forall n. KnownNat (n + 1) => FixedPoint -> Vec (n + 1) FixedPoint -> Vec (n + 1) FixedPoint
softmaxF t xs =
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

-- categorical sampling from FixedPoint probabilities
sampleFromProbsF :: forall n. (KnownNat (n + 1), KnownNat n) => FixedPoint -> Vec (n + 1) FixedPoint -> Unsigned 32
sampleFromProbsF u probs =
  let cdf = CV.scanl1 (+) probs
      idxM = findIndex (>= u) cdf
  in fromIntegral (fromEnum (fromMaybe maxBound idxM))

-- Top-level sampler in FixedPoint
sampledTokenSignal :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Token
  -> TransformerLayer.TransformerDecoderComponent
  -> Signal dom IntermediateData
  -> Signal dom Token
sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal =
  liftA3
    (\temperature logits u ->
        if temperature <= 0 then argMax logits
        else let probabilities = softmaxF temperature logits
             in sampleFromProbsF u probabilities)
    temperatureSignal (logitsSignalF decoder nextIntermediateDataSignal) (uniformRandom01SignalF readyPulseSignal seedSignal)
