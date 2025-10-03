module Model.Layers.Attention.MultiHeadAttention.Internal (
  applyRotaryPositionEncoding
  , computeHeadQ
  , computeHeadKV
  , applyRotation
) where

import Clash.Prelude
import Model.Core.Types
    ( CArray2D(..),
      RotaryEncodingComponent(..),
      RotaryEncodingComponent(..),
      CArray2D(..) )
import Model.Config
    ( ModelDimension,
      HeadDimension,
      RotaryPositionalEmbeddingDimension,
      SequenceLength,
      ModelDimension,
      HeadDimension,
      RotaryPositionalEmbeddingDimension,
      SequenceLength  )
import Model.Numeric.Types ( FixedPoint, FixedPoint )
import Model.Layers.Components.Quantized
    ( SingleHeadComponentQ, SingleHeadComponentQ(..) )
import Model.Helpers.MatVecI8E (matrixVectorMult)

applyRotaryPositionEncoding :: Vec HeadDimension FixedPoint
                             -> Vec RotaryPositionalEmbeddingDimension FixedPoint
                             -> Vec RotaryPositionalEmbeddingDimension FixedPoint
                             -> Vec HeadDimension FixedPoint
applyRotaryPositionEncoding inputVec cosVecF sinVecF =
  concat (imap rotatePair (unconcat d2 inputVec))
  where
    rotatePair :: Index RotaryPositionalEmbeddingDimension -> Vec 2 FixedPoint -> Vec 2 FixedPoint
    rotatePair i vec =
      case vec of
        (realComponent :> imagComponent :> Nil) ->
          let c = cosVecF !! i
              s = sinVecF !! i
              rotatedReal = realComponent * c - imagComponent * s
              rotatedImag = realComponent * s + imagComponent * c
          in rotatedReal :> rotatedImag :> Nil
        _ -> error "Unexpected vector structure in rotatePair"

computeHeadQ
  :: SingleHeadComponentQ
  -> Index SequenceLength
  -> Vec ModelDimension FixedPoint
  -> Vec HeadDimension FixedPoint
computeHeadQ headComp stepCount xHat =
  let q   = matrixVectorMult (wqHeadQ headComp) xHat
      qRo = applyRotation (rotaryQ headComp) stepCount q
  in qRo

computeHeadKV
  :: SingleHeadComponentQ
  -> Index SequenceLength
  -> Vec ModelDimension FixedPoint
  -> (Vec HeadDimension FixedPoint, Vec HeadDimension FixedPoint)
computeHeadKV headComp stepCount xHat =
  let k   = matrixVectorMult (wkHeadQ headComp) xHat
      v   = matrixVectorMult (wvHeadQ headComp) xHat
      kRo = applyRotation (rotaryQ headComp) stepCount k
  in (kRo, v)

applyRotation
  :: RotaryEncodingComponent
  -> Index SequenceLength
  -> Vec HeadDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotation rot stepCount tokenVec =
  let CArray2D arrCos = freqCos rot
      CArray2D arrSin = freqSin rot
      cosF = map realToFrac (arrCos !! stepCount) :: Vec RotaryPositionalEmbeddingDimension FixedPoint
      sinF = map realToFrac (arrSin !! stepCount) :: Vec RotaryPositionalEmbeddingDimension FixedPoint
  in applyRotaryPositionEncoding tokenVec cosF sinF
