module LLaMa2.Sampling.Sampler (
    tokenSampler
) where

import Clash.Prelude
import LLaMa2.Core.Types
  ( Temperature, Token, Seed)
import LLaMa2.Config ( VocabularySize )

import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Sampling.Distribution (pickSample, uniformRandom01Generator)

-- Pure sampling function (combinational)
tokenSampler :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool          -- ^ logitsValid
  -> Signal dom Temperature
  -> Signal dom Seed
  -> Signal dom (Vec VocabularySize FixedPoint)  -- ^ logits
  -> Signal dom Token
tokenSampler logitsValid temperature seed logits = pickSample <$> temperature <*> logits <*> uniformRandom01Generator logitsValid seed
