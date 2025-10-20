module LLaMa2.Layer.Attention.QKVProjection (
  qkvProjectionController
) where

import Clash.Prelude
import LLaMa2.Config
    ( NumQueryHeads,
      ModelDimension,
      NumKeyValueHeads,
      HeadDimension,
      SequenceLength,
      ModelDimension,
      HeadDimension,
      SequenceLength )

import LLaMa2.Numeric.Types ( FixedPoint, FixedPoint )
import LLaMa2.Layer.Components.Quantized
    ( MultiHeadAttentionComponentQ(..),
      SingleHeadComponentQ(..) )
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Helpers.MatVecI8E (matrixMultiplier)
import LLaMa2.Core.Types (ProcessingState (..), LayerData (..))
import LLaMa2.Layer.Attention (fsmController)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)

qkvProjector :: forall dom .
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
qkvProjector validIn readyIn mhaQ seqPosSig xSig =
  (qkvOut, allValid, allReady)
  where
    xNorm = rmsNormFwFix <$> xSig <*> pure (rmsAttF mhaQ)

    -- Propagate readyIn to all Q-head multipliers
    qResults = map (\headQ -> queryHeadProjector validIn readyIn headQ seqPosSig xNorm)
                   (headsQ mhaQ)

    queryHeadsPerKV = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads
    kvHeadIndices :: Vec NumKeyValueHeads (Index NumQueryHeads)
    kvHeadIndices = map (\i -> toEnum (fromEnum i * queryHeadsPerKV)) indicesI

    -- Propagate readyIn to all KV-head multipliers
    kvResults = map (\kvIdx -> let headQ = headsQ mhaQ !! kvIdx
                                in keyValueHeadProjector validIn readyIn headQ seqPosSig xNorm)
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

qkvProjectionController ::
  HiddenClockResetEnable dom =>
  Signal dom Bool -> Signal dom Bool ->
  Signal dom LayerData ->
  MultiHeadAttentionComponentQ ->
  Signal dom ProcessingState ->
  ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  , Signal dom Bool
  , Signal dom Bool )
qkvProjectionController inValid outReady idSig mhaQ psSig = (result, validOut, inReady)
 where
  (enable, validOut, inReady) = fsmController inValid outReady matVecValid
  (result, matVecValid, _ready) =
    qkvProjector enable (pure True) mhaQ
               (sequencePosition <$> psSig)
               (inputVector <$> idSig)

queryHeadProjector
  :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool              -- ^ validIn
  -> Signal dom Bool              -- ^ readyIn (downstream ready)
  -> SingleHeadComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool            -- ^ validOut
     , Signal dom Bool            -- ^ readyOut (can accept input)
     )
queryHeadProjector validIn readyIn headComp stepCountSig xHatSig =
  (qRoOut, validOut, readyOut)
  where
    -- Matrix multiply with handshaking
    (qOut, qValidOut, qReadyOut) =
      matrixMultiplier validIn (pure True) (wqHeadQ headComp) xHatSig

    -- Apply rotary encoding (combinational, but gated by valid)
    qRoOut = (rotaryEncoder (rotaryQ headComp) <$> stepCountSig) <*> qOut

    -- Pass through handshaking signals
    validOut = qValidOut
    readyOut = qReadyOut

keyValueHeadProjector
  :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool              -- ^ validIn
  -> Signal dom Bool              -- ^ readyIn (downstream ready)
  -> SingleHeadComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool            -- ^ validOut
     , Signal dom Bool            -- ^ readyOut (can accept input)
     )
keyValueHeadProjector validIn readyIn headComp stepCountSig xHatSig =
  (kRoOut, vOut, validOut, readyOut)
  where
    -- K matrix multiply
    (kOut, kValidOut, kReadyOut) =
      matrixMultiplier validIn (pure True) (wkHeadQ headComp) xHatSig

    -- V matrix multiply (runs in parallel with K)
    (vOut, vValidOut, vReadyOut) =
      matrixMultiplier validIn (pure True) (wvHeadQ headComp) xHatSig

    -- Apply rotary to K
    kRoOut = (rotaryEncoder (rotaryQ headComp) <$> stepCountSig) <*> kOut

    -- Both K and V must be valid (should happen simultaneously since same input)
    validOut = kValidOut .&&. vValidOut

    -- Can accept input only when both multipliers are ready
    readyOut = kReadyOut .&&. vReadyOut
