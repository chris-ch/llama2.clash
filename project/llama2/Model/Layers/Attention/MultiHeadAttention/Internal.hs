{-# OPTIONS_GHC -Wno-incomplete-patterns #-}
module Model.Layers.Attention.MultiHeadAttention.Internal (
  applyRotaryPositionEncoding
  , computeHeadQ
  , computeHeadKV
  , applyRotation
) where

import Clash.Prelude
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
import Model.Layers.Components.RotaryQ (RotaryEncodingComponentF (..))

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

applyRotaryPositionEncoding
  :: Vec HeadDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotaryPositionEncoding inputVec cosVecF sinVecF =
  concat (imap rotatePair (unconcat d2 inputVec))
 where
  rotatePair :: Index RotaryPositionalEmbeddingDimension -> Vec 2 FixedPoint -> Vec 2 FixedPoint
  rotatePair i (realC :> imagC :> Nil) =
    let c = cosVecF !! i
        s = sinVecF !! i
        r = realC * c - imagC * s
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
