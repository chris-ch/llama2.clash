{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use fewer imports" #-}
module LLaMa2.Layers.Attention.MultiHeadAttention (
  projectQKV
) where

import Clash.Prelude
import LLaMa2.Config
  ( NumQueryHeads, ModelDimension, NumKeyValueHeads
  , HeadDimension, SequenceLength)

import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layers.Components.Quantized
  ( MultiHeadAttentionComponentQ(..) )
import LLaMa2.Layers.Attention.MultiHeadAttention.Internal
  ( computeHeadQ, computeHeadKV )
import LLaMa2.Helpers.FixedPoint (rmsNormFwFix)

-- Quantized MHA: normalize with FixedPoint RMS, compute Q/K/V using I8E mats,
-- and apply rotary inside the per-head kernels.
projectQKV
  :: MultiHeadAttentionComponentQ
  -> Index SequenceLength
  -> Vec ModelDimension FixedPoint
  -> ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint) )
projectQKV mha stepCount inputVector =
  let
    normalizedInput = rmsNormFwFix inputVector (rmsAttF mha)

    queries =
      imap (\qIx _ ->
              let headQ = headsQ mha !! qIx
              in computeHeadQ headQ stepCount normalizedInput)
           indicesI

    keysAndValues =
      imap (\kvIx _ ->
        let nQ  = natToNum @NumQueryHeads :: Int
            nKV = natToNum @NumKeyValueHeads :: Int
            qIdx0 = toEnum (min (nQ - 1) (fromEnum kvIx * (nQ `div` nKV))) :: Index NumQueryHeads
            headQ = headsQ mha !! qIdx0
        in computeHeadKV headQ stepCount normalizedInput)
      indicesI

    (keys, values) = unzip keysAndValues
  in (queries, keys, values)
