module LLaMa2.Layer.Attention.KVCache (
    kvBankController
) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, NumQueryHeads, HeadDimension, NumKeyValueHeads, SequenceLength)
import LLaMa2.Types.LayerData (ProcessingState (..), LayerData (..), CycleStage (..))
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.KVCacheBank as Cache
import LLaMa2.Memory.DualPortRAM (trueDualPortRam)
import LLaMa2.Layer.Attention.AttentionHead (attentionHead)

kvBankController ::
  forall dom.
  HiddenClockResetEnable dom =>
  Index NumLayers ->
  Signal dom (Index SequenceLength) ->      -- seqPos (not ProcessingState)
  Signal dom LayerData ->
  Signal dom Bool ->
  Signal dom Bool ->  -- enableWriteKV
  Signal dom Bool ->  -- enableAttend
  ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
  , Vec NumQueryHeads (Signal dom Bool)
  , Vec NumKeyValueHeads (Signal dom Bool) ) ->
  Index NumKeyValueHeads ->
  ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
  , Vec NumQueryHeads (Signal dom Bool)
  , Vec NumKeyValueHeads (Signal dom Bool) )
kvBankController layerIndex seqPos layerData qkvValid 
                 enableWriteKV enableAttend
                 (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  (headOutAcc2, headDoneAcc2, writeDoneAcc1)
 where
  -- Use enable signals directly (already done in Step 2)
  isStage2Write = enableWriteKV
  isStage3Attn = enableAttend

  -- Query indices
  qIdx0 = queryHeadIndex0 kvIx
  hasQ1 = hasSecondQueryHead kvIx
  qIdx1 = queryHeadIndex1 kvIx

  query0 = getQueryVector layerData qIdx0
  query1 = if hasQ1 then getQueryVector layerData qIdx1 else pure (repeat 0)
  keyVec = getKeyVector layerData kvIx
  valVec = getValueVector layerData kvIx

  -- KV Write controller
  (wrPulse, wrDone) = Cache.writePulseGenerator (isStage2Write .&&. qkvValid)
  wrKVRowK = mux wrPulse (Just <$> bundle (seqPos, keyVec)) (pure Nothing)
  wrKVRowV = mux wrPulse (Just <$> bundle (seqPos, valVec)) (pure Nothing)

  (kRow, kRowB) = trueDualPortRam tAddrRow (pure Nothing) seqPos wrKVRowK
  (vRow, vRowB) = trueDualPortRam tAddrRow (pure Nothing) seqPos wrKVRowV
  writeDoneAcc1 = replace kvIx wrDone writeDoneAcc

  -- Attention row sequencer
  attnPrev = register False isStage3Attn
  clearS3  = liftA2 (\now prev -> now && not prev) isStage3Attn attnPrev
  (tAddrRow, stepEnRow, lastTRow) = attentionRowSequencer clearS3 isStage3Attn seqPos

  -- Per-head attention
  (out0, done0) = attentionHead clearS3 stepEnRow query0 kRow vRow lastTRow
  (out1, done1) = if hasQ1 then attentionHead clearS3 stepEnRow query1 kRow vRow lastTRow
                           else (pure (repeat 0), pure False)

  headOutAcc1  = replace qIdx0 out0 headOutAcc
  headOutAcc2  = if hasQ1 then replace qIdx1 out1 headOutAcc1 else headOutAcc1

  headDoneAcc1 = replace qIdx0 done0 headDoneAcc
  headDoneAcc2 = if hasQ1 then replace qIdx1 done1 headDoneAcc1 else headDoneAcc1

attentionRowSequencer ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> ( Signal dom (Index SequenceLength)
     , Signal dom Bool
     , Signal dom Bool )
attentionRowSequencer clearS3 isStage3Attention seqPosSignal =
  let
    rowCounter :: Signal dom (Index SequenceLength)
    rowCounter = mealy rowCounterT 0 (bundle (clearS3, isStage3Attention, seqPosSignal))

    rowCounterT :: Index SequenceLength -> (Bool, Bool, Index SequenceLength)
                -> (Index SequenceLength, Index SequenceLength)
    rowCounterT t (clearPulse, stageActive, pos) =
      let
        tStart = if clearPulse then 0 else t
        step   = stageActive
        isLast   = step && tStart == pos
        tNext  = if not step || isLast then tStart else succ tStart
      in (tNext, tStart)

    stepNow :: Signal dom Bool
    stepNow = const <$> isStage3Attention <*> rowCounter

    stepEnRow :: Signal dom Bool
    stepEnRow = register False stepNow

    lastNow :: Signal dom Bool
    lastNow = (==) <$> rowCounter <*> seqPosSignal

    lastTRow :: Signal dom Bool
    lastTRow = register False lastNow

  in (rowCounter, stepEnRow, lastTRow)

queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 kvIx = toEnum (min maxQueryHeadIndex (baseQueryIndex kvIx))

queryHeadIndex1 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex1 kvIx =
  if hasSecondQueryHead kvIx
    then toEnum (baseQueryIndex kvIx + 1)
    else queryHeadIndex0 kvIx

getQueryVector :: Signal dom LayerData -> Index NumQueryHeads -> Signal dom (Vec HeadDimension FixedPoint)
getQueryVector idSig qIx = (\i -> queryVectors i !! qIx) <$> idSig

getKeyVector :: Signal dom LayerData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension FixedPoint)
getKeyVector idSig kvIx = (\i -> keyVectors i !! kvIx) <$> idSig

getValueVector :: Signal dom LayerData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension FixedPoint)
getValueVector idSig kvIx = (\i -> valueVectors i !! kvIx) <$> idSig

queryHeadsPerKeyValueHead :: Int
queryHeadsPerKeyValueHead = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

maxQueryHeadIndex :: Int
maxQueryHeadIndex = natToNum @NumQueryHeads - 1

baseQueryIndex :: Index NumKeyValueHeads -> Int
baseQueryIndex kvIx = fromEnum kvIx * queryHeadsPerKeyValueHead

hasSecondQueryHead :: Index NumKeyValueHeads -> Bool
hasSecondQueryHead kvIx =
  queryHeadsPerKeyValueHead >= 2 && (baseQueryIndex kvIx + 1 <= maxQueryHeadIndex)
