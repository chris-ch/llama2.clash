module Model.Layers.Attention.MultiHeadAttention.Internal where

import Clash.Prelude

import Model.Core.Types (NumQueryHeads, ModelDimemsion, NumKeyValueHeads,
  HeadDimension, CArray2D (..), SingleHeadComponent (..),
  RotaryPositionalEmbeddingDimension, RotaryEncodingComponent (..), SequenceLength)
import Helpers (matrixVectorMult, rmsNorm)
import qualified Prelude as P

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  -- | Per-head output projection matrix W_O (shape HeadDim × ModelDimemsion)
  , mWo :: Vec NumQueryHeads (CArray2D ModelDimemsion HeadDimension)
  -- | RMSNorm before QKV projection (size ModelDimemsion)
  , rmsAtt :: Vec ModelDimemsion Float
  } deriving (Show)

applyRotaryPositionEncoding :: Vec HeadDimension Float
  -> Vec RotaryPositionalEmbeddingDimension Float
  -> Vec RotaryPositionalEmbeddingDimension Float
  -> Vec HeadDimension Float
applyRotaryPositionEncoding inputVec cosVec sinVec =
  concat (imap rotatePair (unconcat d2 inputVec))
 where
  rotatePair :: Index RotaryPositionalEmbeddingDimension -> Vec 2 Float -> Vec 2 Float
  rotatePair i (realComponent :> imagComponent :> Nil) =
    let c = cosVec !! i
        s = sinVec !! i
        rotatedReal = realComponent * c - imagComponent * s
        rotatedImag = realComponent * s + imagComponent * c
    in  rotatedReal :> rotatedImag :> Nil

-- Apply rotation per head
applyRotation :: RotaryEncodingComponent
  -> Index SequenceLength
  -> Vec HeadDimension Float
  -> Vec HeadDimension Float
applyRotation rot stepCount tokenVec =
  let
    CArray2D arrFreqCos = freqCos rot
    CArray2D arrFreqSin = freqSin rot
    cosFrequencies = arrFreqCos !! stepCount
    sinFrequencies = arrFreqSin !! stepCount
  in applyRotaryPositionEncoding tokenVec cosFrequencies sinFrequencies

-- Compute K/V for a head
computeHeadKV
  :: SingleHeadComponent
  -> Index SequenceLength
  -> Vec ModelDimemsion Float
  -> (Vec HeadDimension Float, Vec HeadDimension Float)
computeHeadKV headComp stepCount xHat =
  let
    k = matrixVectorMult (wkHead headComp) xHat  -- HeadDimension x ModelDimemsion * ModelDimemsion -> HeadDimension
    v = matrixVectorMult (wvHead headComp) xHat  -- HeadDimension x ModelDimemsion * ModelDimemsion -> HeadDimension
    kRot = applyRotation (rotary headComp) stepCount k
    CArray2D _wQ = wqHead headComp
    CArray2D _wK = wkHead headComp
    CArray2D _wV = wvHead headComp
  in (kRot, v)

-- Compute Q for a head
computeHeadQ
  :: SingleHeadComponent
  -> Index SequenceLength
  -> Vec ModelDimemsion Float
  -> Vec HeadDimension Float
computeHeadQ headComp stepCount xHat =
  let
    q = matrixVectorMult (wqHead headComp) xHat  -- HeadDimension x ModelDimemsion * ModelDimemsion -> HeadDimension
    qRot = applyRotation (rotary headComp) stepCount q

    CArray2D _wK = wkHead headComp
  in qRot
