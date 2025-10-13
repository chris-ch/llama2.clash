module LLaMa2.Layers.Attention.MultiHeadAttention (
  projectQKV, projectQKVSeq
) where

import Clash.Prelude
import LLaMa2.Config
  ( NumQueryHeads, ModelDimension, NumKeyValueHeads
  , HeadDimension, SequenceLength)

import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layers.Components.Quantized
  ( MultiHeadAttentionComponentQ(..) )
import LLaMa2.Layers.Attention.MultiHeadAttention.Internal
  ( computeHeadQ, computeHeadKV, computeHeadQSeq, computeHeadKVSeq )
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

projectQKVSeq :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool              -- ^ validIn (enable computation)
  -> Signal dom Bool              -- ^ readyIn (downstream ready - for protocol)
  -> MultiHeadAttentionComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool            -- ^ validOut (all heads done)
     , Signal dom Bool            -- ^ readyOut (can accept - always True in this design)
     )
projectQKVSeq validIn readyIn mhaQ seqPosSig xSig =
  (qkvOut, allValid, allReady)
  where
    -- Normalize input once
    xNorm = liftA2 rmsNormFwFix xSig (pure $ rmsAttF mhaQ)
    
    -- Project all query heads (runs when validIn)
    qResults = map (\headQ -> computeHeadQSeq validIn (pure True) headQ seqPosSig xNorm)
                   (headsQ mhaQ)
    
    -- KV heads (grouped query attention pattern)
    queryHeadsPerKV = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads
    kvHeadIndices :: Vec NumKeyValueHeads (Index NumQueryHeads)
    kvHeadIndices = map (\i -> toEnum (fromEnum i * queryHeadsPerKV)) indicesI
    
    kvResults = map (\kvIdx -> let headQ = headsQ mhaQ !! kvIdx
                                in computeHeadKVSeq validIn (pure True) headQ seqPosSig xNorm)
                    kvHeadIndices
    
    -- Unpack results
    qVecs    = map (\(q, _, _) -> q) qResults
    qValids  = map (\(_, v, _) -> v) qResults
    qReadys  = map (\(_, _, r) -> r) qResults
    
    kVecs    = map (\(k, _, _, _) -> k) kvResults
    vVecs    = map (\(_, v, _, _) -> v) kvResults
    kvValids = map (\(_, _, v, _) -> v) kvResults
    kvReadys = map (\(_, _, _, r) -> r) kvResults
    
    -- KEY: All must be valid for output to be valid
    allValid = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
    
    -- All are ready when idle (simplified - they're always ready in this design)
    allReady = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)
    
    -- Bundle outputs
    qkvOut = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)
