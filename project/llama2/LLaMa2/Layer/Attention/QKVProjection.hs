module LLaMa2.Layer.Attention.QKVProjection
  (
    keyValueHeadProjector
  , qkvProjectionController
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
  ( NumQueryHeads, NumKeyValueHeads
  , ModelDimension, HeadDimension, SequenceLength
  )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified LLaMa2.Layer.Attention.FSM as FSM (processingControllerFSM)
import qualified Simulation.Parameters as PARAM (MultiHeadAttentionComponentQ(..), SingleHeadComponentQ(..))
import qualified LLaMa2.Layer.Attention.QKVProjectionWeightBuffer as WEIGHTS
  ( QKVProjectionWeightBuffer(..)
  , extractQWeight, extractKWeight, extractVWeight
  )
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplierDyn)

--------------------------------------------------------------------------------
-- Q head projector with weight selection (hardcoded vs RAM)
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM weights
  -> PARAM.SingleHeadComponentQ               -- ^ hardcoded (fallback)
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     )
queryHeadProjector inputValid downStreamReady stepCountSig xHatSig ramQ headParams =
  (qRoOut, outputValid, readyForInput)
 where
  selectedQ :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedQ = pure (PARAM.wqHeadQ headParams) -- should be ramQ

  (qOut, outputValid, readyForInput) =
    parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedQ xHatSig
  
  qRoOut = (rotaryEncoder (PARAM.rotaryF headParams) <$> stepCountSig) <*> qOut

--------------------------------------------------------------------------------
-- KV head projector with weight selection (hardcoded vs RAM)
--------------------------------------------------------------------------------
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM K
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM V
  -> PARAM.SingleHeadComponentQ               -- ^ hardcoded (fallback)
  -> ( Signal dom (Vec HeadDimension FixedPoint)  -- K
     , Signal dom (Vec HeadDimension FixedPoint)  -- V
     , Signal dom Bool                            -- outputValid
     , Signal dom Bool                            -- readyForInput
     )
keyValueHeadProjector inputValid downStreamReady stepCountSig xHatSig ramK ramV headParams =
  (kRoOut, vOut, outputValid, readyForInput)
 where
  selectedK :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedK = pure (PARAM.wkHeadQ headParams) -- should be ramK

  selectedV :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedV = pure (PARAM.wvHeadQ headParams) -- should be ramV

  -- Use downStreamReady for both children
  (kOut, kValidOut, kReadyOut) =
    parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedV xHatSig

  kRoOut = (rotaryEncoder (PARAM.rotaryF headParams) <$> stepCountSig) <*> kOut

  -- Keep existing policy: present result only when both valid, and upstream is ready when both children are ready
  outputValid = kValidOut .&&. vValidOut
  readyForInput = kReadyOut .&&. vReadyOut

--------------------------------------------------------------------------------
-- Full QKV projector using the RAM buffer
--------------------------------------------------------------------------------
qkvProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom WEIGHTS.QKVProjectionWeightBuffer               -- ^ RAM weight buffer
  -> PARAM.MultiHeadAttentionComponentQ             -- ^ hardcoded (fallback)
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     )
qkvProjector inputValid downStreamReady seqPosSig xSig weightBuffer mhaParams =
  (qkvOut, outputValid, readyForInput)
 where
  xNorm = rmsNormFwFix <$> xSig <*> pure (PARAM.rmsAttF mhaParams)
  -- Q heads
  qResults = imap qHead (PARAM.headsQ mhaParams)
   where
    qHead :: Index NumQueryHeads -> PARAM.SingleHeadComponentQ
          -> ( Signal dom (Vec HeadDimension FixedPoint)
             , Signal dom Bool
             , Signal dom Bool )
    qHead headIx headQ =
      let ramQ = WEIGHTS.extractQWeight <$> weightBuffer <*> pure headIx
      in queryHeadProjector inputValid downStreamReady seqPosSig xNorm ramQ headQ

  -- Map KV heads to their corresponding SingleHeadComponentQ (for rotary params)
  queryHeadsPerKV = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads
  kvHeadIndices :: Vec NumKeyValueHeads (Index NumQueryHeads)
  kvHeadIndices = map (\i -> toEnum (fromEnum i * queryHeadsPerKV)) indicesI

  kvResults = imap kvHead kvHeadIndices
   where
    kvHead :: Index NumKeyValueHeads -> Index NumQueryHeads
           -> ( Signal dom (Vec HeadDimension FixedPoint)
              , Signal dom (Vec HeadDimension FixedPoint)
              , Signal dom Bool
              , Signal dom Bool )
    kvHead kvIx qIx =
      let headParams = PARAM.headsQ mhaParams !! qIx
          ramK  = WEIGHTS.extractKWeight <$> weightBuffer <*> pure kvIx
          ramV  = WEIGHTS.extractVWeight <$> weightBuffer <*> pure kvIx
      in keyValueHeadProjector inputValid downStreamReady seqPosSig xNorm ramK ramV headParams

  qVecs    = map (\(q, _, _) -> q) qResults
  qValids  = map (\(_, v, _) -> v) qResults
  qReadys  = map (\(_, _, r) -> r) qResults

  kVecs    = map (\(k, _, _, _) -> k) kvResults
  vVecs    = map (\(_, v, _, _) -> v) kvResults
  kvValids = map (\(_, _, v, _) -> v) kvResults
  kvReadys = map (\(_, _, _, r) -> r) kvResults

  outputValid = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
  readyForInput = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)

  qkvOut = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)

--------------------------------------------------------------------------------
-- Controller wrapper
--------------------------------------------------------------------------------
qkvProjectionController ::
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.MultiHeadAttentionComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom WEIGHTS.QKVProjectionWeightBuffer
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool )
qkvProjectionController inputValid downStreamReady input mhaParams seqPos weightsBuffer =
  (result, outputValid, readyForInput)
 where
  -- Generic controller (stateful): produces raw enable/valid/inReady
  (enableRaw, outputValid, inReadyRaw) =
    FSM.processingControllerFSM inputValid downStreamReady matVecValid

  -- Instantiate the projector, now driving downstream readyIn correctly.
  -- We also capture its upstream readiness (projReadyOut).
  (result, matVecValid, projReadyOut) =
    qkvProjector enableGated downStreamReady
                 seqPos
                 input
                 weightsBuffer
                 mhaParams

  -- Gate enable and inReady with a one-cycle delayed projector readiness
  -- to avoid any potential combinational loop; start permissive (True).
  projReadyOut_d = register True projReadyOut
  enableGated    = enableRaw  .&&. projReadyOut_d
  readyForInput   = inReadyRaw .&&. projReadyOut_d
