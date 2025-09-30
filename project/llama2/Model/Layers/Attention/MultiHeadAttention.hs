{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use fewer imports" #-}
module Model.Layers.Attention.MultiHeadAttention (
  projectQKV
) where

import Clash.Prelude
import qualified Prelude as P
import Model.Core.Types
  ( NumQueryHeads, ModelDimemsion, NumKeyValueHeads
  , HeadDimension, CArray2D (..), SingleHeadComponent (..)
  , SequenceLength)

import Clash.Prelude
import qualified Prelude as P


import Clash.Prelude

import Model.Core.Types
  ( NumQueryHeads, NumKeyValueHeads, HeadDimension
  , ModelDimemsion, SequenceLength )
import Model.Numeric.Types (FixedPoint)
import Model.Layers.Components.Quantized
  ( MultiHeadAttentionComponentQ(..), SingleHeadComponentQ(..) )
import Model.Layers.Attention.MultiHeadAttention.Internal
  ( computeHeadQF_Q, computeHeadKVF_Q )
import Model.Helpers.FixedPoint (rmsNormFwFix)

-- Quantized MHA: normalize with FixedPoint RMS, compute Q/K/V using I8E mats,
-- and apply rotary inside the per-head kernels.
projectQKV
  :: MultiHeadAttentionComponentQ
  -> Index SequenceLength
  -> Vec ModelDimemsion FixedPoint
  -> ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint) )
projectQKV mha stepCount inputVector =
  let
    normalizedInput = rmsNormFwFix inputVector (rmsAttF mha)

    queries =
      imap (\qIx _ ->
              let headQ = headsQ mha !! qIx
              in computeHeadQF_Q headQ stepCount normalizedInput)
           indicesI

    keysAndValues =
      imap (\kvIx _ ->
        let nQ  = natToNum @NumQueryHeads :: Int
            nKV = natToNum @NumKeyValueHeads :: Int
            qIdx0 = toEnum (min (nQ - 1) (fromEnum kvIx * (nQ `div` nKV))) :: Index NumQueryHeads
            headQ = headsQ mha !! qIdx0
        in computeHeadKVF_Q headQ stepCount normalizedInput)
      indicesI

    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
