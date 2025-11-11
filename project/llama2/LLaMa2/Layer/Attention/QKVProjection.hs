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
import LLaMa2.Layer.Attention.FSM (processingControllerFSM)
import qualified Simulation.Parameters as PARAM (MultiHeadAttentionComponentQ(..), SingleHeadComponentQ(..))
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer
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
  -> PARAM.SingleHeadComponentQ               -- ^ hardcoded (fallback)
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM weights
  -> Signal dom Bool                          -- ^ useRAM
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     )
queryHeadProjector inputValid downStreamReady headComp stepCountSig xHatSig ramWeights useRAM =
  (qRoOut, outputValid, readyForInput)
 where
  selectedMat :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedMat = mux useRAM ramWeights (pure (PARAM.wqHeadQ headComp))

  (qOut, outputValid, readyForInput) =
    parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedMat xHatSig
  
  qRoOut = (rotaryEncoder (PARAM.rotaryF headComp) <$> stepCountSig) <*> qOut

--------------------------------------------------------------------------------
-- KV head projector with weight selection (hardcoded vs RAM)
--------------------------------------------------------------------------------
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> PARAM.SingleHeadComponentQ               -- ^ hardcoded (fallback)
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM K
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM V
  -> Signal dom Bool                          -- ^ useRAM
  -> ( Signal dom (Vec HeadDimension FixedPoint)  -- K
     , Signal dom (Vec HeadDimension FixedPoint)  -- V
     , Signal dom Bool                            -- outputValid
     , Signal dom Bool                            -- readyForInput
     )
keyValueHeadProjector inputValid downStreamReady headComp stepCountSig xHatSig ramK ramV useRAM =
  (kRoOut, vOut, outputValid, readyForInput)
 where
  selectedK = mux useRAM ramK (pure (PARAM.wkHeadQ headComp))
  selectedV = mux useRAM ramV (pure (PARAM.wvHeadQ headComp))

  -- Use downStreamReady for both children
  (kOut, kValidOut, kReadyOut) =
    parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedV xHatSig

  kRoOut = (rotaryEncoder (PARAM.rotaryF headComp) <$> stepCountSig) <*> kOut

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
  -> PARAM.MultiHeadAttentionComponentQ             -- ^ hardcoded (fallback)
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom QKVProjectionWeightBuffer               -- ^ RAM weight buffer
  -> Signal dom Bool                          -- ^ useRAM (fully loaded)
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     )
qkvProjector inputValid downStreamReady mhaQ seqPosSig xSig weightBuffer useRAM =
  (qkvOut, outputValid, readyForInput)
 where
  xNorm = rmsNormFwFix <$> xSig <*> pure (PARAM.rmsAttF mhaQ)
  useRAM' = useRAM -- should be pure True for disabling legacy wired weights completely for QKV
  -- Q heads
  qResults = imap qHead (PARAM.headsQ mhaQ)
   where
    qHead :: Index NumQueryHeads -> PARAM.SingleHeadComponentQ
          -> ( Signal dom (Vec HeadDimension FixedPoint)
             , Signal dom Bool
             , Signal dom Bool )
    qHead hIx headQ =
      let ramQ = extractQWeight <$> weightBuffer <*> pure hIx
      in queryHeadProjector inputValid downStreamReady headQ seqPosSig xNorm ramQ useRAM'

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
      let headQ = PARAM.headsQ mhaQ !! qIx
          ramK  = extractKWeight <$> weightBuffer <*> pure kvIx
          ramV  = extractVWeight <$> weightBuffer <*> pure kvIx
      in keyValueHeadProjector inputValid downStreamReady headQ seqPosSig xNorm ramK ramV useRAM'

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
  -> Signal dom QKVProjectionWeightBuffer
  -> Signal dom Bool
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool )
qkvProjectionController inputValid downStreamReady input mhaQ seqPos weightBuf useRAM =
  (result, outputValid, readyForInput)
 where
  -- Generic controller (stateful): produces raw enable/valid/inReady
  (enableRaw, outputValid, inReadyRaw) =
    processingControllerFSM inputValid downStreamReady matVecValid

  -- Instantiate the projector, now driving downstream readyIn correctly.
  -- We also capture its upstream readiness (projReadyOut).
  (result, matVecValid, projReadyOut) =
    qkvProjector enableGated downStreamReady mhaQ
                 seqPos
                 input
                 weightBuf
                 useRAM

  -- Gate enable and inReady with a one-cycle delayed projector readiness
  -- to avoid any potential combinational loop; start permissive (True).
  projReadyOut_d = register True projReadyOut
  enableGated    = enableRaw  .&&. projReadyOut_d
  readyForInput   = inReadyRaw .&&. projReadyOut_d
