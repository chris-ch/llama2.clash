module Model.Layers.Attention.MultiHeadAttention.Internal where

import Clash.Prelude
import qualified Prelude as P
import Model.Core.Types
  ( NumQueryHeads, ModelDimemsion, NumKeyValueHeads
  , HeadDimension, CArray2D (..), SingleHeadComponent (..)
  , RotaryPositionalEmbeddingDimension, RotaryEncodingComponent (..)
  , SequenceLength)
import Model.Numeric.Types (FixedPoint)
import Model.Helpers.Fixed (matrixVectorMultF)
import Helpers (matrixVectorMult)

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

applyRotationF :: RotaryEncodingComponent
  -> Index SequenceLength
  -> Vec HeadDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotationF rot stepCount tokenVec =
  let
    CArray2D arrFreqCos = freqCos rot   -- Float params
    CArray2D arrFreqSin = freqSin rot
    cosFrequencies = map realToFrac (arrFreqCos !! stepCount) :: Vec RotaryPositionalEmbeddingDimension FixedPoint
    sinFrequencies = map realToFrac (arrFreqSin !! stepCount) :: Vec RotaryPositionalEmbeddingDimension FixedPoint
  in applyRotaryPositionEncodingF tokenVec cosFrequencies sinFrequencies

computeHeadKVF
  :: SingleHeadComponent
  -> Index SequenceLength
  -> Vec ModelDimemsion FixedPoint
  -> (Vec HeadDimension FixedPoint, Vec HeadDimension FixedPoint)
computeHeadKVF headComp stepCount xHat =
  let
    k = matrixVectorMultF (wkHead headComp) xHat
    v = matrixVectorMultF (wvHead headComp) xHat
    kRot = applyRotationF (rotary headComp) stepCount k
  in (kRot, v)

computeHeadQF
  :: SingleHeadComponent
  -> Index SequenceLength
  -> Vec ModelDimemsion FixedPoint
  -> Vec HeadDimension FixedPoint
computeHeadQF headComp stepCount xHat =
  let
    q = matrixVectorMultF (wqHead headComp) xHat
    qRot = applyRotationF (rotary headComp) stepCount q
  in qRot
