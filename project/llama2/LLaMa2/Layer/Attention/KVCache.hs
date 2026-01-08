module LLaMa2.Layer.Attention.KVCache (
    kvBankController
) where

import Clash.Prelude
import qualified Prelude as P
import LLaMa2.Types.ModelConfig (NumQueryHeads, HeadDimension, NumKeyValueHeads, SequenceLength)
import LLaMa2.Types.LayerData (LayerData (..))
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.KVCacheBank as Cache
import LLaMa2.Memory.DualPortRAM (trueDualPortRam)
import LLaMa2.Layer.Attention.AttentionHead (attentionHead)
import TraceUtils (traceEdgeC, traceChangeC)
import Clash.Debug (trace)

kvBankController ::
  forall dom.
  HiddenClockResetEnable dom =>
  Signal dom (Unsigned 32) ->              -- cycleCounter for tracing
  Signal dom (Index SequenceLength) ->
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
kvBankController cycleCounter seqPos layerData qkvValid 
                 enableWriteKV enableAttend
                 (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  (headOutAcc2, headDoneAcc2, writeDoneAcc1)
 where

  -- Query indices
  qIdx0 = queryHeadIndex0 kvIx
  hasQ1 = hasSecondQueryHead kvIx
  qIdx1 = queryHeadIndex1 kvIx

  query0 = getQueryVector layerData qIdx0
  query1 = if hasQ1 then getQueryVector layerData qIdx1 else pure (repeat 0)
  keyVec = getKeyVector layerData kvIx
  valVec = getValueVector layerData kvIx

  -- KV Write controller
  (wrPulse, wrDone) = Cache.writePulseGenerator (enableWriteKV .&&. qkvValid)
  wrKVRowK = mux wrPulse (Just <$> bundle (seqPos, keyVec)) (pure Nothing)
  wrKVRowV = mux wrPulse (Just <$> bundle (seqPos, valVec)) (pure Nothing)

  -- Trace KV writes - couple to wrDone to force evaluation
  wrPulseTraced = traceKVWrite cycleCounter kvIx wrPulse seqPos keyVec valVec
  wrDoneTraced = liftA2 const wrDone wrPulseTraced

  (kRow, _kRowB) = trueDualPortRam tAddrRow (pure Nothing) seqPos wrKVRowK
  (vRow, _vRowB) = trueDualPortRam tAddrRow (pure Nothing) seqPos wrKVRowV
  writeDoneAcc1 = replace kvIx wrDoneTraced writeDoneAcc  -- Use traced version

  -- Attention row sequencer
  attnPrev = register False enableAttend
  clearS3  = liftA2 (\now prev -> now && not prev) enableAttend attnPrev
  (tAddrRow, stepEnRow, lastTRow) = attentionRowSequencer cycleCounter kvIx clearS3 enableAttend seqPos

  -- Trace KV reads (when stepping through attention) - couple to kRow to force evaluation
  kRowTraced = forceTrace <$> kRow <*> traceKVRead cycleCounter kvIx stepEnRow tAddrRow kRow
    where forceTrace k () = k

  -- Per-head attention (use kRowTraced to ensure trace evaluates)
  (out0, done0) = attentionHead clearS3 stepEnRow query0 kRowTraced vRow lastTRow
  (out1, done1) = if hasQ1 then attentionHead clearS3 stepEnRow query1 kRowTraced vRow lastTRow
                           else (pure (repeat 0), pure False)

  headOutAcc1  = replace qIdx0 out0 headOutAcc
  headOutAcc2  = if hasQ1 then replace qIdx1 out1 headOutAcc1 else headOutAcc1

  headDoneAcc1 = replace qIdx0 done0 headDoneAcc
  headDoneAcc2 = if hasQ1 then replace qIdx1 done1 headDoneAcc1 else headDoneAcc1

-- | Trace KV cache writes
traceKVWrite :: 
  Signal dom (Unsigned 32) ->
  Index NumKeyValueHeads ->
  Signal dom Bool ->
  Signal dom (Index SequenceLength) ->
  Signal dom (Vec HeadDimension FixedPoint) ->
  Signal dom (Vec HeadDimension FixedPoint) ->
  Signal dom Bool
traceKVWrite cyc kvIx wrPulse seqPos keyVec valVec = result
  where
    result = emit <$> cyc <*> wrPulse <*> seqPos <*> keyVec <*> valVec
    emit c True pos kv vv = 
      trace ("@" P.++ show c P.++ " [KVC KV" P.++ show kvIx P.++ "] WRITE pos=" P.++ show pos 
             P.++ " K[0]=" P.++ show (head kv) P.++ " V[0]=" P.++ show (head vv)) True
    emit _ False _ _ _ = False

-- | Trace KV cache reads during attention
traceKVRead ::
  Signal dom (Unsigned 32) ->
  Index NumKeyValueHeads ->
  Signal dom Bool ->
  Signal dom (Index SequenceLength) ->
  Signal dom (Vec HeadDimension FixedPoint) ->
  Signal dom ()
traceKVRead cyc kvIx stepEn readAddr kRow = result
  where
    result = emit <$> cyc <*> stepEn <*> readAddr <*> kRow
    emit c True addr kv = 
      trace ("@" P.++ show c P.++ " [KVC KV" P.++ show kvIx P.++ "] READ pos=" P.++ show addr 
             P.++ " K[0]=" P.++ show (head kv)) ()
    emit _ False _ _ = ()

attentionRowSequencer ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)       -- cycleCounter
  -> Index NumKeyValueHeads         -- kvIx for tracing
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> ( Signal dom (Index SequenceLength)
     , Signal dom Bool
     , Signal dom Bool )
attentionRowSequencer cycleCounter kvIx clearS3 enableAttend seqPosSignal =
  let
    rowCounter :: Signal dom (Index SequenceLength)
    rowCounter = mealy rowCounterT 0 (bundle (clearS3, enableAttend, seqPosSignal))

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
    stepNow = const <$> enableAttend <*> rowCounter

    stepEnRow :: Signal dom Bool
    stepEnRow = register False stepNow

    lastNow :: Signal dom Bool
    lastNow = (==) <$> rowCounter <*> seqPosSignal

    lastTRow :: Signal dom Bool
    lastTRow = register False lastNow

    -- Trace row counter changes and key events
    rowCounterTraced = traceChangeC cycleCounter ("[KVC KV" P.++ show kvIx P.++ "] rowCounter") rowCounter
    lastTRowTraced = traceEdgeC cycleCounter ("[KVC KV" P.++ show kvIx P.++ "] lastTRow") lastTRow

    -- Trace attention start - force evaluation by coupling to stepEnRow
    clearS3Traced = traceEdgeC cycleCounter ("[KVC KV" P.++ show kvIx P.++ "] attnStart") clearS3
    stepEnRowTraced = liftA2 const stepEnRow clearS3Traced  -- Forces clearS3Traced evaluation

  in (rowCounterTraced, stepEnRowTraced, lastTRowTraced)

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
