module Model.Core.Embedding
  ( embed ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint, scalePow2F, Act, ExpS)
import Model.Numeric.ParamPack (QArray2D(..))
import Model.Core.Types (Token, ModelDimemsion, VocabularySize)

-- Dequantize-on-read: mant * 2^exp -> FixedPoint vector
embed
  :: QArray2D VocabularySize ModelDimemsion
  -> Token
  -> Vec ModelDimemsion FixedPoint
embed (QArray2D table) tok =
  let (mant, e) = table !! (fromIntegral tok :: Int)
      s         = scalePow2F e 1
  in map (\q -> fromIntegral q * s) mant
