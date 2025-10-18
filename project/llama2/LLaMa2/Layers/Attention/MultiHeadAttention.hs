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


projectQKV :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool              -- ^ validIn (enable computation)
  -> Signal dom Bool              -- ^ readyIn (downstream ready)
  -> MultiHeadAttentionComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool            -- ^ validOut (all heads done)
     , Signal dom Bool            -- ^ readyOut (can accept)
     )
projectQKV validIn readyIn mhaQ seqPosSig xSig =
  (qkvOut, allValid, allReady)
  where
    xNorm = rmsNormFwFix <$> xSig <*> pure (rmsAttF mhaQ)

    -- Propagate readyIn to all Q-head multipliers
    qResults = map (\headQ -> computeHeadQ validIn readyIn headQ seqPosSig xNorm)
                   (headsQ mhaQ)

    queryHeadsPerKV = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads
    kvHeadIndices :: Vec NumKeyValueHeads (Index NumQueryHeads)
    kvHeadIndices = map (\i -> toEnum (fromEnum i * queryHeadsPerKV)) indicesI

    -- Propagate readyIn to all KV-head multipliers
    kvResults = map (\kvIdx -> let headQ = headsQ mhaQ !! kvIdx
                                in computeHeadKV validIn readyIn headQ seqPosSig xNorm)
                    kvHeadIndices

    qVecs    = map (\(q, _, _) -> q) qResults
    qValids  = map (\(_, v, _) -> v) qResults
    qReadys  = map (\(_, _, r) -> r) qResults

    kVecs    = map (\(k, _, _, _) -> k) kvResults
    vVecs    = map (\(_, v, _, _) -> v) kvResults
    kvValids = map (\(_, _, v, _) -> v) kvResults
    kvReadys = map (\(_, _, _, r) -> r) kvResults

    allValid = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
    allReady = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)

    qkvOut = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)
