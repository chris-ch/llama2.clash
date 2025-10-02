module Model.Layers.TransformerLayer (
    multiCycleTransformerLayer
  , TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
) where

import Clash.Prelude
import Model.Core.Types
  ( ProcessingState(..), LayerData(..), CycleStage(..)
  )
import Model.Config
  ( ModelDimension
  , NumLayers, NumQueryHeads, NumKeyValueHeads
  , HeadDimension,  SequenceLength
  )
import qualified Model.Memory.KVCacheBank as Cache

import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention (projectQKV)
import Model.Layers.Components.Quantized
  ( FeedForwardNetworkComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , EmbeddingComponentQ(..)
  )

import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork (computeFeedForward)
import Model.Helpers.MatVecI8E (matrixVectorMult)
import Model.Numeric.Types (FixedPoint)
import Helpers (liftA4)
import Model.Layers.Attention.AttentionHead (attendHead)
import Model.Memory.KVCacheBank.RowStreamer (kvRowStreamer)
import Model.Layers.Attention.AttendSequential (attendHeadSeq)
import Model.Config.Debug (AttnMode(..), attnMode, attnEps)
import Model.Memory.RamOps (toRamOperation)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttentionComponentQ
  , feedforwardNetwork :: FeedForwardNetworkComponentQ
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponentQ
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

multiCycleTransformerLayer ::
  forall dom .
  HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Cache.KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom LayerData
  -> ( Signal dom LayerData
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom LayerData
  )
multiCycleTransformerLayer layer kvRamOwner layerIndex processingState layerData =
  ( nextLayerData
  , writeDone
  , attentionDone
  , commitStage3
  )
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

  (perHeadOutputs, perHeadDoneFlags, perBankWriteDoneFlags) =
    let initHeadOutputs = repeat (pure (repeat 0))
        initHeadDone    = repeat (pure False)
        initWriteDone   = repeat (pure False)
    in  foldl
          (fillOneBank layerIndex processingState kvRamOwner layerData)
          (initHeadOutputs, initHeadDone, initWriteDone)
          indicesI

  allHeadsDone :: Signal dom Bool
  allHeadsDone = and <$> sequenceA perHeadDoneFlags

  allHeadsDonePrev :: Signal dom Bool
  allHeadsDonePrev = register False allHeadsDone

  attentionDone :: Signal dom Bool
  attentionDone =
    liftA2 (\now prev -> now && not prev) allHeadsDone allHeadsDonePrev

  baseNextLayerData :: Signal dom LayerData
  baseNextLayerData =
    liftA2 (stageProcessor mha ffn layerIndex) processingState layerData

  allBanksDone :: Signal dom Bool
  allBanksDone = and <$> sequenceA perBankWriteDoneFlags

  writeDone :: Signal dom Bool
  writeDone = kvWriteDoneCond layerIndex <$> processingState <*> allBanksDone

  -- Per-head WO projection uses quantized WO blocks (I8E) -> FixedPoint
  perHeadProjectedSignalsVec :: Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint))
  perHeadProjectedSignalsVec =
    zipWith (\woQ hSig -> matrixVectorMult woQ <$> hSig) (mWoQ mha) perHeadOutputs

  perHeadProjected :: Signal dom (Vec NumQueryHeads (Vec ModelDimension FixedPoint))
  perHeadProjected = sequenceA perHeadProjectedSignalsVec

  woHeads :: Signal dom (Vec ModelDimension FixedPoint)
  woHeads    = foldl1 (zipWith (+)) <$> perHeadProjected

  xAfterAttn :: Signal dom (Vec ModelDimension FixedPoint)
  xAfterAttn = liftA2 inputsAggregator layerData woHeads

  nextLayerData :: Signal dom LayerData
  nextLayerData = liftA4 (layerDataAttnDone layerIndex) processingState baseNextLayerData xAfterAttn attentionDone

  commitStage3 :: Signal dom LayerData
  commitStage3 = liftA4 (layerDataAttnDone layerIndex)  processingState layerData xAfterAttn attentionDone

kvWriteDoneCond :: Index NumLayers -> ProcessingState -> Bool -> Bool
kvWriteDoneCond layerIndex state banksDone = processingStage state == Stage2_WriteKV
      && processingLayer state == layerIndex
      && banksDone

inputsAggregator :: LayerData -> Vec ModelDimension FixedPoint -> Vec ModelDimension FixedPoint
inputsAggregator layerData = zipWith (+) (inputVector layerData)

layerDataAttnDone :: Index NumLayers
  -> ProcessingState
  -> LayerData
  -> Vec ModelDimension FixedPoint
  -> Bool
  -> LayerData
layerDataAttnDone layerIndex stage cur attOut attnDone =
        if processingLayer stage == layerIndex
          && processingStage stage == Stage3_Attend
          && attnDone
          then cur { attentionOutput = attOut }
          else cur

stageProcessor :: MultiHeadAttentionComponentQ
  -> FeedForwardNetworkComponentQ
  -> Index NumLayers
  -> ProcessingState
  -> LayerData
  -> LayerData
stageProcessor mhaQ ffnQ layerIndex ps idata
  | processingLayer ps /= layerIndex = idata
  | otherwise = case processingStage ps of
      Stage1_ProjectQKV ->
        let
          (qs, ks, vs) = MultiHeadAttention.projectQKV mhaQ (sequencePosition ps) (inputVector idata)
        in idata { queryVectors = qs, keyVectors = ks, valueVectors = vs }

      Stage2_WriteKV     -> idata
      Stage3_Attend      -> idata

      Stage4_FeedForward ->
        let ffnOut = FeedForwardNetwork.computeFeedForward ffnQ (attentionOutput idata)
        in  idata { feedForwardOutput = ffnOut }

      Stage5_Bookkeeping -> idata

-- Query heads per KV head
queryHeadsPerKeyValueHead :: Int
queryHeadsPerKeyValueHead = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

maxQueryHeadIndex :: Int
maxQueryHeadIndex = natToNum @NumQueryHeads - 1

baseQueryIndex :: Index NumKeyValueHeads -> Int
baseQueryIndex kvIx = fromEnum kvIx * queryHeadsPerKeyValueHead

queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 kvIx = toEnum (min maxQueryHeadIndex (baseQueryIndex kvIx))

hasSecondQueryHead :: Index NumKeyValueHeads -> Bool
hasSecondQueryHead kvIx = queryHeadsPerKeyValueHead >= 2 && (baseQueryIndex kvIx + 1 <= maxQueryHeadIndex)

queryHeadIndex1 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex1 kvIx =
  if hasSecondQueryHead kvIx then toEnum (baseQueryIndex kvIx + 1) else queryHeadIndex0 kvIx

getQueryVector :: Signal dom LayerData -> Index NumQueryHeads -> Signal dom (Vec HeadDimension FixedPoint)
getQueryVector idSig qIx = (\i -> queryVectors i !! qIx) <$> idSig

getKeyVector :: Signal dom LayerData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension FixedPoint)
getKeyVector idSig kvIx = (\i -> keyVectors i !! kvIx) <$> idSig

getValueVector :: Signal dom LayerData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension FixedPoint)
getValueVector idSig kvIx = (\i -> valueVectors i !! kvIx) <$> idSig

fillOneBank ::
  forall dom .
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Cache.KVRamOwner dom
  -> Signal dom LayerData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
fillOneBank layerIndex psSig kvOwner idSig (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  let
    -- Stage predicates for this layer
    stageEquals :: CycleStage -> Signal dom Bool
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIndex)
             psSig (pure ())
    
    isStage3Attention :: Signal dom Bool
    isStage3Attention = stageEquals Stage3_Attend

    isStage2Write :: Signal dom Bool
    isStage2Write     = stageEquals Stage2_WriteKV

    -- Stage3 entry pulse
    attnPrev :: Signal dom Bool
    attnPrev = register False isStage3Attention

    clearS3 :: Signal dom Bool
    clearS3  = liftA2 (\now prev -> now && not prev) isStage3Attention attnPrev

    seqPosSignal :: Signal dom (Index SequenceLength)
    seqPosSignal = sequencePosition <$> psSig

    -- Query head mapping
    qIdx0 = queryHeadIndex0 kvIx
    hasQ1 = hasSecondQueryHead kvIx
    qIdx1 = queryHeadIndex1 kvIx

    query0 :: Signal dom (Vec HeadDimension FixedPoint)
    query0 = getQueryVector idSig qIdx0

    query1 :: Signal dom (Vec HeadDimension FixedPoint)
    query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)

    keyVec :: Signal dom (Vec HeadDimension FixedPoint)
    keyVec   = getKeyVector   idSig kvIx      -- FixedPoint vector (one row)

    valueVec :: Signal dom (Vec HeadDimension FixedPoint)
    valueVec = getValueVector idSig kvIx

    -- ========== Stage2: writes ==========
    -- Quantized write to the I8E KV bank
    (writeAddrSig, kMantWrRaw, kExpWrRaw, vMantWrRaw, vExpWrRaw, writeDoneThisBank) = Cache.writeSequencer isStage2Write seqPosSignal (bundle (keyVec, valueVec))
    wrAddrS = mux isStage2Write writeAddrSig (pure 0)
    kMantWr = mux isStage2Write kMantWrRaw (pure Nothing)
    vMantWr = mux isStage2Write vMantWrRaw (pure Nothing)
    kExpWr  = mux isStage2Write kExpWrRaw  (pure Nothing)
    vExpWr  = mux isStage2Write vExpWrRaw  (pure Nothing)

    (tAddrRow, stepEnRow, lastTRow) = attentionRowSequencer clearS3 isStage3Attention seqPosSignal

    -- Build operations for the two row BRAMs
    wrKVRowK :: Signal dom (Maybe (Index SequenceLength, Vec HeadDimension FixedPoint))
    wrKVRowK =
      mux isStage2Write
          (Just <$> bundle (seqPosSignal, keyVec))
          (pure Nothing)

    wrKVRowV :: Signal dom (Maybe (Index SequenceLength, Vec HeadDimension FixedPoint))
    wrKVRowV =
      mux isStage2Write
          (Just <$> bundle (seqPosSignal, valueVec))
          (pure Nothing)

    -- Instantiate the two BRAMs
    (kRowA, _kRowB) =
      trueDualPortBlockRam
        (toRamOperation tAddrRow (pure Nothing))
        (toRamOperation seqPosSignal wrKVRowK)

    (vRowA, _vRowB) =
      trueDualPortBlockRam
        (toRamOperation tAddrRow (pure Nothing))
        (toRamOperation seqPosSignal wrKVRowV)

    -- ========== BASELINE: combinational attend over register mirror ==========
    kvKeysAll :: Signal dom (Vec SequenceLength (Vec HeadDimension FixedPoint))
    kvKeysAll = mealy
      (\mem (we, p, rowK) ->
         let mem' = if we then replace p rowK mem else mem
         in  (mem', mem'))
      (repeat (repeat 0))
      (bundle (isStage2Write, seqPosSignal, keyVec))

    kvValsAll :: Signal dom (Vec SequenceLength (Vec HeadDimension FixedPoint))
    kvValsAll = mealy
      (\mem (we, p, rowV) ->
         let mem' = if we then replace p rowV mem else mem
         in  (mem', mem'))
      (repeat (repeat 0))
      (bundle (isStage2Write, seqPosSignal, valueVec))

    out0_baseline = liftA4 attendHead query0 kvKeysAll kvValsAll seqPosSignal
    out1_baseline =
      if hasQ1
        then liftA4 attendHead query1 kvKeysAll kvValsAll seqPosSignal
        else pure (repeat 0)
    doneBaseline = clearS3

    -- ========== STREAMED (quantized BRAM, I8E -> F) for shadow/compare ==========
    bank :: Cache.KVBank dom
    bank = Cache.kvBanks kvOwner !! kvIx

    (kRowQ, vRowQ, rowValidQ, lastTQ) =
      kvRowStreamer bank clearS3 isStage3Attention seqPosSignal
                    wrAddrS kMantWr kExpWr vMantWr vExpWr
    (out0_seqQ, done0_seqQ) = attendHeadSeq clearS3 rowValidQ query0 kRowQ vRowQ lastTQ
    (out1_seqQ, done1_seqQ) =
      if hasQ1
        then attendHeadSeq clearS3 rowValidQ query1 kRowQ vRowQ lastTQ
        else (pure (repeat 0), pure False)

    -- ========== STREAMED (unquantized BRAM rows, FixedPoint) for progressive replacement ==========
    (out0_seqF, done0_seqF) = attendHeadSeq clearS3 stepEnRow query0 kRowA vRowA lastTRow
    (out1_seqF, done1_seqF) =
      if hasQ1
        then attendHeadSeq clearS3 stepEnRow query1 kRowA vRowA lastTRow
        else (pure (repeat 0), pure False)

    -- Selection
    (out0_sel, out1_sel, done0_sel, done1_sel) =
      case attnMode of
        AttnBaseline      -> (out0_baseline, out1_baseline, doneBaseline, doneBaseline)
        AttnShadowBRAM    -> (out0_baseline, out1_baseline, doneBaseline, doneBaseline)
        AttnReplaceBRAMF -> (out0_seqF,     out1_seqF,     done0_seqF,   done1_seqF)
        AttnReplaceBRAMQ -> (out0_seqQ,     out1_seqQ,     done0_seqQ,   done1_seqQ)

    -- Optional diagnostics (safe to keep; low footprint)

    -- Row error monitors (max-abs diff per row)
    maxAbs v = foldl max 0 (map abs v)

    diffTooBig = liftA2
        (\a b -> maxAbs (zipWith (-) a b) > attnEps)

    kRowErr  = diffTooBig kRowQ kRowA
    vRowErr  = diffTooBig vRowQ vRowA

    -- Latch last errors at lastTQ (end of stream)
    lastKRowErr = regEn (False :: Bool) lastTQ kRowErr
    lastVRowErr = regEn (False :: Bool) lastTQ vRowErr

    !_probeK = lastKRowErr
    !_probeV = lastVRowErr


    !_bramQCnt0 = mealy (\c m -> let c' = if m then c+1 else c in (c', c')) (0 :: Unsigned 16)
                        (diffTooBig out0_baseline out0_seqQ)
    !_bramQCnt1 = mealy (\c m -> let c' = if m then c+1 else c in (c', c')) (0 :: Unsigned 16)
                        (if hasQ1 then diffTooBig out1_baseline out1_seqQ else pure False)

    -- Accumulate into return vectors
    headOutAcc0  = replace qIdx0 out0_sel headOutAcc

    headOutAcc1 :: Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
    headOutAcc1  = if hasQ1 then replace qIdx1 out1_sel headOutAcc0 else headOutAcc0

    headDoneAcc0 = replace qIdx0 done0_sel headDoneAcc

    headDoneAcc1 :: Vec NumQueryHeads (Signal dom Bool)
    headDoneAcc1 = if hasQ1 then replace qIdx1 done1_sel headDoneAcc0 else headDoneAcc0

    writeDoneAcc1 :: Vec NumKeyValueHeads (Signal dom Bool)
    writeDoneAcc1 = replace kvIx writeDoneThisBank writeDoneAcc

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1)

-- | Generate row read addresses and step/last signals for progressive replacement
-- Unquantized row BRAMs (FixedPoint) used only for progressive replacement:
-- Two true-dual-port BRAMs with (depth = SequenceLength, payload = Vec HeadDimension FixedPoint)
-- Port A: Stage3 reads; Port B: Stage2 writes current row at (seqPos)
-- Address/Write streams for BRAM-F
-- | Generate row read addresses and step/last signals for progressive replacement
attentionRowSequencer ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool                  -- ^ clearS3 (stage 3 entry pulse)
  -> Signal dom Bool                  -- ^ isStage3Attention
  -> Signal dom (Index SequenceLength) -- ^ seqPosSignal
  -> ( Signal dom (Index SequenceLength)  -- tAddrRow
     , Signal dom Bool                   -- stepEnRow
     , Signal dom Bool )                 -- lastTRow
attentionRowSequencer clearS3 isStage3Attention seqPosSignal =
  let
    -- === Step 1: manage row counter ===
    -- Reset to 0 on Stage3 entry
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

    -- === Step 2: detect step enable ===
    stepNow :: Signal dom Bool
    stepNow = liftA2 const isStage3Attention rowCounter

    stepEnRow :: Signal dom Bool
    stepEnRow = register False stepNow

    -- === Step 3: detect last row ===
    lastNow :: Signal dom Bool
    lastNow = liftA2 (==) rowCounter seqPosSignal

    lastTRow :: Signal dom Bool
    lastTRow = register False lastNow

  in (rowCounter, stepEnRow, lastTRow)
