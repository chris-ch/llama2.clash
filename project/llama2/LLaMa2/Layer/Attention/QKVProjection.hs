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
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import LLaMa2.Layer.Attention.FSM (processingControllerFSM)
import qualified Simulation.Parameters as PARAM (MultiHeadAttentionComponentQ(..), SingleHeadComponentQ(..))
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer
  ( QKVProjectionWeightBuffer(..)
  , extractQWeight, extractKWeight, extractVWeight
  )
-- Reuse the proven row core and control FSM
import LLaMa2.Numeric.Operations
  ( parallel64RowProcessor
  , matrixMultiplierStateMachine, MultiplierState (..)
  )

--------------------------------------------------------------------------------
-- Dynamic matrix-vector multiplier: accepts Signal dom (MatI8E rows cols)
-- Mirrors parallel64RowMatrixMultiplier but with runtime-selectable matrix.
--------------------------------------------------------------------------------
parallelRowMatrixMultiplierDyn :: forall dom rows cols.
  ( HiddenClockResetEnable dom
  , KnownNat rows, KnownNat cols
  )
  => Signal dom Bool                      -- ^ validIn
  -> Signal dom Bool                      -- ^ readyIn (downstream)
  -> Signal dom (MatI8E rows cols)        -- ^ matrix (runtime, from RAM or const)
  -> Signal dom (Vec cols FixedPoint)     -- ^ input vector
  -> ( Signal dom (Vec rows FixedPoint)   -- ^ output vector
     , Signal dom Bool                    -- ^ validOut
     , Signal dom Bool                    -- ^ readyOut
     )
parallelRowMatrixMultiplierDyn validIn readyInDownstream matSig inputVector =
  (outputVector, validOut, readyOut)
 where
  -- Row counter
  rowIndex :: Signal dom (Index rows)
  rowIndex = register 0 nextRowIndex

  -- Fetch current row from the runtime matrix
  currentRow :: Signal dom (RowI8E cols)
  currentRow = (!!) <$> matSig <*> rowIndex

  -- Parallel row engine
  (rowResult, rowDone) =
    parallel64RowProcessor rowReset rowEnable currentRow inputVector

  -- Protocol FSM
  (state, rowReset, rowEnable, validOut, readyOut) =
    matrixMultiplierStateMachine validIn readyInDownstream rowDone rowIndex

  -- Row index sequencing
  nextRowIndex =
    mux (rowDone .&&. (rowIndex ./=. pure maxBound))
        (rowIndex + 1)
        (mux ((state .==. pure MDone) .&&. readyInDownstream)
             (pure 0)
             rowIndex)

  -- Accumulate per-row results into the output vector
  outputVector = register (repeat 0) nextOutput
  nextOutput   = mux rowDone
                   (replace <$> rowIndex <*> rowResult <*> outputVector)
                   outputVector

--------------------------------------------------------------------------------
-- Q head projector with weight selection (hardcoded vs RAM)
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                          -- ^ validIn
  -> Signal dom Bool                          -- ^ readyIn
  -> PARAM.SingleHeadComponentQ                     -- ^ hardcoded (fallback)
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM weights
  -> Signal dom Bool                          -- ^ useRAM
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool                        -- ^ validOut
     , Signal dom Bool                        -- ^ readyOut
     )
queryHeadProjector validIn readyIn headComp stepCountSig xHatSig ramWeights useRAM =
  (qRoOut, validOut, readyOut)
 where
  -- Select matrix dynamically
  selectedMat :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedMat = mux useRAM ramWeights (pure (PARAM.wqHeadQ headComp))

  -- MatVec with dynamic matrix
  (qOut, qValidOut, qReadyOut) =
    parallelRowMatrixMultiplierDyn validIn (pure True) selectedMat xHatSig

  -- Rotary encoding on Q
  qRoOut = (rotaryEncoder (PARAM.rotaryF headComp) <$> stepCountSig) <*> qOut

  validOut = qValidOut
  readyOut = qReadyOut

--------------------------------------------------------------------------------
-- KV head projector with weight selection (hardcoded vs RAM)
--------------------------------------------------------------------------------
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                          -- ^ validIn
  -> Signal dom Bool                          -- ^ readyIn
  -> PARAM.SingleHeadComponentQ                     -- ^ hardcoded (fallback)
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM K
  -> Signal dom (MatI8E HeadDimension ModelDimension)  -- ^ RAM V
  -> Signal dom Bool                          -- ^ useRAM
  -> ( Signal dom (Vec HeadDimension FixedPoint)  -- K
     , Signal dom (Vec HeadDimension FixedPoint)  -- V
     , Signal dom Bool                            -- validOut
     , Signal dom Bool                            -- readyOut
     )
keyValueHeadProjector validIn readyIn headComp stepCountSig xHatSig ramK ramV useRAM =
  (kRoOut, vOut, validOut, readyOut)
 where
  selectedK = mux useRAM ramK (pure (PARAM.wkHeadQ headComp))
  selectedV = mux useRAM ramV (pure (PARAM.wvHeadQ headComp))

  (kOut, kValidOut, kReadyOut) =
    parallelRowMatrixMultiplierDyn validIn (pure True) selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    parallelRowMatrixMultiplierDyn validIn (pure True) selectedV xHatSig

  kRoOut = (rotaryEncoder (PARAM.rotaryF headComp) <$> stepCountSig) <*> kOut

  validOut = kValidOut .&&. vValidOut
  readyOut = kReadyOut .&&. vReadyOut

--------------------------------------------------------------------------------
-- Full QKV projector using the RAM buffer
--------------------------------------------------------------------------------
qkvProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                          -- ^ validIn
  -> Signal dom Bool                          -- ^ readyIn
  -> PARAM.MultiHeadAttentionComponentQ             -- ^ hardcoded (fallback)
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom QKVProjectionWeightBuffer               -- ^ RAM weight buffer
  -> Signal dom Bool                          -- ^ useRAM (fully loaded)
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool                        -- ^ validOut
     , Signal dom Bool                        -- ^ readyOut
     )
qkvProjector validIn readyIn mhaQ seqPosSig xSig weightBuffer useRAM =
  (qkvOut, allValid, allReady)
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
      in queryHeadProjector validIn readyIn headQ seqPosSig xNorm ramQ useRAM'

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
      in keyValueHeadProjector validIn readyIn headQ seqPosSig xNorm ramK ramV useRAM'

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

--------------------------------------------------------------------------------
-- Controller wrapper (unchanged protocol; adds RAM buffer + useRAM)
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
qkvProjectionController validIn readyIn input mhaQ seqPos weightBuf useRAM =
  (result, validOut, inReady)
 where
  (projectorEnable, validOut, inReady) =
    processingControllerFSM validIn readyIn matVecValid

  (result, matVecValid, ready) =
    qkvProjector projectorEnable (pure True) mhaQ
                 seqPos
                 input
                 weightBuf
                 useRAM
