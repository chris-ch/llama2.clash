module LLaMa2.Core.Embedding
  ( embedder ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.ParamPack (QArray2D(..))
import LLaMa2.Core.Types (Token)
import LLaMa2.Config (LLaMa2Dimension, VocabularySize)

-- BRAM/ROM-backed dequantize-on-read. 1-cycle latency.
embedder
  :: HiddenClockResetEnable dom
  => QArray2D VocabularySize LLaMa2Dimension
  -> Signal dom Token
  -> Signal dom (Vec LLaMa2Dimension FixedPoint)
embedder (QArray2D table) tokSig =
  let
    -- Precompute dequantized rows at elaboration time; stored in ROM.
    deqRow (mant, e) =
      let s = scalePow2F e 1
      in map (\q -> fromIntegral q * s) mant
    romContent :: Vec VocabularySize (Vec LLaMa2Dimension FixedPoint)
    romContent = map deqRow table
  in rom romContent tokSig
