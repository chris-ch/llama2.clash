module LLaMa2.Layers.Attention.MultiHeadAttention.Internal (
  applyRotaryPositionEncoding
  , computeHeadQ
  , computeHeadKV
  , applyRotation
) where

import Clash.Prelude
import LLaMa2.Config
    ( ModelDimension,
      HeadDimension,
      RotaryPositionalEmbeddingDimension,
      SequenceLength,
      ModelDimension,
      HeadDimension,
      RotaryPositionalEmbeddingDimension,
      SequenceLength  )
import LLaMa2.Numeric.Types ( FixedPoint, FixedPoint )
import LLaMa2.Layers.Components.Quantized
    ( SingleHeadComponentQ, SingleHeadComponentQ(..) )
import LLaMa2.Layers.Components.RotaryQ (RotaryEncodingComponentF (..))
import LLaMa2.Helpers.MatVecI8E (matrixMultiplier)

computeHeadQ
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool                              -- validIn
  -> Signal dom Bool                              -- readyIn (downstream)
  -> SingleHeadComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec HeadDimension FixedPoint)  -- qRo result
     , Signal dom Bool                            -- validOut
     , Signal dom Bool                            -- readyOut
     )
computeHeadQ validIn readyInDownstream headComp stepCount xHat =
  let (q, validQ, readyQ) = matrixMultiplier validIn readyInDownstream (wqHeadQ headComp) xHat
      qRo = applyRotation (rotaryQ headComp) <$> stepCount <*> q
  in (qRo, validQ, readyQ)

computeHeadKV
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool                              -- validIn
  -> Signal dom Bool                              -- readyIn (downstream)
  -> SingleHeadComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec HeadDimension FixedPoint)  -- kRo result
     , Signal dom (Vec HeadDimension FixedPoint)  -- v result
     , Signal dom Bool                            -- validOut
     , Signal dom Bool                            -- readyOut
     )
computeHeadKV validIn readyInDownstream headComp stepCount xHat =
  let -- K and V in parallel
      (k, validK, readyK) = matrixMultiplier validIn readyInDownstream (wkHeadQ headComp) xHat
      (v, validV, _)      = matrixMultiplier validIn readyInDownstream (wvHeadQ headComp) xHat
      
      kRo = applyRotation (rotaryQ headComp) <$> stepCount <*> k
      
      -- Both must be valid
      validOut = validK .&&. validV
      
  in (kRo, v, validOut, readyK)

-- helper: safely destructure a Vec 2
vec2ToPair :: NFDataX a => Vec 2 a -> (a, a)
vec2ToPair (x :> y :> Nil) = (x, y)
vec2ToPair _   = deepErrorX "Impossible: Vec 2 had wrong shape"

applyRotaryPositionEncoding
  :: Vec HeadDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotaryPositionEncoding inputVec cosVecF sinVecF =
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

applyRotation
  :: RotaryEncodingComponentF
  -> Index SequenceLength
  -> Vec HeadDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotation rot step tokenVec =
  let cosF = freqCosF rot !! step
      sinF = freqSinF rot !! step
  in  applyRotaryPositionEncoding tokenVec cosF sinF
