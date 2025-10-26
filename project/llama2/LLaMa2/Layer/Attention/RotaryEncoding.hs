module LLaMa2.Layer.Attention.RotaryEncoding
  ( rotaryEncoder
  ) where
    
import Clash.Prelude

import LLaMa2.Types.ModelConfig 
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM (RotaryEncodingComponentF (..))

rotaryEncoder
  :: PARAM.RotaryEncodingComponentF
  -> Index SequenceLength
  -> Vec HeadDimension FixedPoint
  -> Vec HeadDimension FixedPoint
rotaryEncoder rot step tokenVec =
  let cosF = PARAM.freqCosF rot !! step
      sinF = PARAM.freqSinF rot !! step
  in  rotaryPositionEncoder tokenVec cosF sinF

-- Helper: safely destructure a Vec 2
vec2ToPair :: NFDataX a => Vec 2 a -> (a, a)
vec2ToPair (x :> y :> Nil) = (x, y)
vec2ToPair _   = deepErrorX "Impossible: Vec 2 had wrong shape"

rotaryPositionEncoder
  :: Vec HeadDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec HeadDimension FixedPoint
rotaryPositionEncoder inputVec cosVecF sinVecF =
  concat (imap rotatePair (unconcat d2 inputVec))
 where
  rotatePair :: Index RotaryPositionalEmbeddingDimension
             -> Vec 2 FixedPoint
             -> Vec 2 FixedPoint
  rotatePair i v =
    let (realC, imagC) = vec2ToPair v
        c  = cosVecF !! i
        s  = sinVecF !! i
        r  = realC * c - imagC * s
        im = realC * s + imagC * c
    in  r :> im :> Nil
