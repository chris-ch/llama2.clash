module LLaMa2.Layers.Attention.MultiHeadAttention (
  projectQKVSeq
) where

import Clash.Prelude
import LLaMa2.Config
  ( NumQueryHeads, ModelDimension, NumKeyValueHeads
  , HeadDimension, SequenceLength)

import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layers.Components.Quantized
  ( MultiHeadAttentionComponentQ(..) )
import LLaMa2.Layers.Attention.MultiHeadAttention.Internal
  ( computeHeadQSeq, computeHeadKVSeq )
import LLaMa2.Helpers.FixedPoint (rmsNormFwFix)

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
