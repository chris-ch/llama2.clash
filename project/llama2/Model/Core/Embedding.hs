module Model.Core.Embedding
  ( embedder ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint, scalePow2F)
import Model.Numeric.ParamPack (QArray2D(..))
import Model.Core.Types (Token)
import Model.Config (ModelDimension, VocabularySize)

-- BRAM/ROM-backed dequantize-on-read. 1-cycle latency.
embedder
  :: HiddenClockResetEnable dom
  => QArray2D VocabularySize ModelDimension
  -> Signal dom Token
  -> Signal dom (Vec ModelDimension FixedPoint)
embedder (QArray2D table) tokSig =
  let
    -- Precompute dequantized rows at elaboration time; stored in ROM.
    deqRow (mant, e) =
      let s = scalePow2F e 1
      in map (\q -> fromIntegral q * s) mant
    romContent :: Vec VocabularySize (Vec ModelDimension FixedPoint)
    romContent = map deqRow table
  in rom romContent tokSig
