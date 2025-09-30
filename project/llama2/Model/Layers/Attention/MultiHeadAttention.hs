module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..), projectQKV
) where

import Clash.Prelude
import qualified Prelude as P
import Model.Core.Types
  ( NumQueryHeads, ModelDimemsion, NumKeyValueHeads
  , HeadDimension, CArray2D (..), SingleHeadComponent (..)
  , SequenceLength)
import Model.Helpers.Fixed (rmsNormF)
import Model.Numeric.Types (FixedPoint)
import Model.Layers.Attention.MultiHeadAttention.Internal
  ( computeHeadKVF
  , computeHeadQF )

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  , mWo    :: Vec NumQueryHeads (CArray2D ModelDimemsion HeadDimension)
  , rmsAtt :: Vec ModelDimemsion Float
  } deriving (Show)

projectQKV :: MultiHeadAttentionComponent
  -> Index SequenceLength
  -> Vec ModelDimemsion FixedPoint
  -> ( Vec NumQueryHeads   (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint) )
projectQKV multiHeadAttentionComponent stepCount inputVector =
  let
    normalizedInput = rmsNormF inputVector (rmsAtt multiHeadAttentionComponent)

    queries =
      imap (\queryHeadIdx _ ->
              let headComponent = heads multiHeadAttentionComponent !! queryHeadIdx
              in computeHeadQF headComponent stepCount normalizedInput)
           indicesI

    keysAndValues =
      imap (\keyValueHeadIdx _ ->
        let qIdx0 = fromEnum keyValueHeadIdx * (natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads)
            queryIndex = toEnum (min (natToNum @NumQueryHeads - 1) qIdx0) :: Index NumQueryHeads
            headComponent = heads multiHeadAttentionComponent !! queryIndex
        in computeHeadKVF headComponent stepCount normalizedInput)
      indicesI

    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
