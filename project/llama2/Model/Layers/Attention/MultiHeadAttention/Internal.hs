module Model.Layers.Attention.MultiHeadAttention.Internal where

import Clash.Prelude
import qualified Prelude as P
import Model.Core.Types
  ( NumQueryHeads, ModelDimemsion, NumKeyValueHeads
  , HeadDimension, CArray2D (..), SingleHeadComponent (..)
  , RotaryPositionalEmbeddingDimension, RotaryEncodingComponent (..)
  , SequenceLength)
import Model.Numeric.Types (FixedPoint)
import Model.Helpers.FixedPoint (matrixVectorMultF)
import Model.Layers.Components.Quantized (SingleHeadComponentQ)
import qualified Prelude as P
import Model.Core.Types
  ( ModelDimemsion, HeadDimension
  , RotaryEncodingComponent(..)
  , RotaryPositionalEmbeddingDimension
  , SequenceLength, CArray2D(..) )
import Model.Numeric.Types (FixedPoint)
import Model.Helpers.MatVecI8E (matrixVectorMultI8E_Fixed)
import Model.Layers.Components.Quantized (SingleHeadComponentQ(..))


applyRotaryPositionEncodingF :: Vec HeadDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotaryPositionEncodingF inputVec cosVecF sinVecF =
  concat (imap rotatePair (unconcat d2 inputVec))
 where
  rotatePair :: Index RotaryPositionalEmbeddingDimension -> Vec 2 FixedPoint -> Vec 2 FixedPoint
  rotatePair i (realComponent :> imagComponent :> Nil) =
    let c = cosVecF !! i
        s = sinVecF !! i
        rotatedReal = realComponent * c - imagComponent * s
        rotatedImag = realComponent * s + imagComponent * c
    in  rotatedReal :> rotatedImag :> Nil

computeHeadQF_Q
  :: SingleHeadComponentQ
  -> Index SequenceLength
  -> Vec ModelDimemsion FixedPoint
  -> Vec HeadDimension FixedPoint
computeHeadQF_Q headComp stepCount xHat =
  let q   = matrixVectorMultI8E_Fixed (wqHeadQ headComp) xHat
      qRo = applyRotationF (rotaryQ headComp) stepCount q
  in qRo

computeHeadKVF_Q
  :: SingleHeadComponentQ
  -> Index SequenceLength
  -> Vec ModelDimemsion FixedPoint
  -> (Vec HeadDimension FixedPoint, Vec HeadDimension FixedPoint)
computeHeadKVF_Q headComp stepCount xHat =
  let k   = matrixVectorMultI8E_Fixed (wkHeadQ headComp) xHat
      v   = matrixVectorMultI8E_Fixed (wvHeadQ headComp) xHat
      kRo = applyRotationF (rotaryQ headComp) stepCount k
  in (kRo, v)

applyRotationF
  :: RotaryEncodingComponent
  -> Index SequenceLength
  -> Vec HeadDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotationF rot stepCount tokenVec =
  let CArray2D arrCos = freqCos rot
      CArray2D arrSin = freqSin rot
      cosF = map realToFrac (arrCos !! stepCount) :: Vec RotaryPositionalEmbeddingDimension FixedPoint
      sinF = map realToFrac (arrSin !! stepCount) :: Vec RotaryPositionalEmbeddingDimension FixedPoint
  in applyRotaryPositionEncodingF tokenVec cosF sinF
