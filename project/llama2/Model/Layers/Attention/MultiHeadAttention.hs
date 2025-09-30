module Model.Layers.Attention.MultiHeadAttention (
    projectQKVq
) where

import Clash.Prelude
import qualified Prelude as P
import Model.Core.Types
  ( NumQueryHeads, ModelDimemsion, NumKeyValueHeads
  , HeadDimension, CArray2D (..), SingleHeadComponent (..)
  , SequenceLength)

import Clash.Prelude
import qualified Prelude as P

import Model.Core.Types
  ( NumQueryHeads, ModelDimemsion, NumKeyValueHeads
  , HeadDimension, SequenceLength
  )
import Model.Numeric.Types (FixedPoint)
import Model.Layers.Components.Quantized
  ( MultiHeadAttentionComponentQ(..), SingleHeadComponentQ(..) )
import Model.Layers.Attention.MultiHeadAttention.Internal
  ( computeHeadKVF_Q
  , computeHeadQF_Q
  )
import Model.Helpers.FixedPoint (rmsNormFwFix)

projectQKVq
  :: MultiHeadAttentionComponentQ
  -> Index SequenceLength
  -> Vec ModelDimemsion FixedPoint
  -> ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint) )
projectQKVq mha stepCount inputVector =
  let
    normalizedInput = rmsNormFwFix inputVector (rmsAttF mha)

    queries =
      imap (\qIx _ ->
              let headComp = headsQ mha !! qIx
              in computeHeadQF_Q headComp stepCount normalizedInput)
           indicesI

    -- map KV head index to a representative query head (same as Float path)
    qHeadsPerKV :: Int
    qHeadsPerKV = natToNum @NumQueryHeads `P.div` natToNum @NumKeyValueHeads

    kvToQIndex :: Index NumKeyValueHeads -> Index NumQueryHeads
    kvToQIndex kvIx =
      let base = fromEnum kvIx * qHeadsPerKV
          capped = min (natToNum @NumQueryHeads - 1) base
      in toEnum capped

    keysAndValues =
      imap (\kvIx _ ->
        let qIx = kvToQIndex kvIx
            headComp = headsQ mha !! qIx
        in computeHeadKVF_Q headComp stepCount normalizedInput)
      indicesI

    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
