module Model.Layers.TransformerLayer (
    multiCycleTransformerLayer
  , TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
) where

import Clash.Prelude
import Model.Core.Types
  ( ModelDimemsion, EmbeddingComponent(..)
  , NumLayers, NumQueryHeads, NumKeyValueHeads
  , HeadDimension, ProcessingState(..), IntermediateData(..), CycleStage(..), SequenceLength
  )
import Model.Helpers.Fixed (rmsNormF, matrixVectorMultF)

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork
import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention

import Data.Maybe (fromMaybe)
import Model.Numeric.Types (ExpS, FixedPoint)
import Helpers (liftA4)
import Model.Layers.Attention.AttentionHead.Fixed (attendHeadF)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttention.MultiHeadAttentionComponent
  , feedforwardNetwork :: FeedForwardNetwork.FeedForwardNetworkComponent
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponent
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

multiCycleTransformerLayer
  :: HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Cache.KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom IntermediateData
  -> ( Signal dom IntermediateData
     , Signal dom Bool                         -- writeDone (Stage2_WriteKV)
     , Signal dom Bool                         -- attnDone  (Stage3_Attend, rising)
     , Signal dom IntermediateData             -- commitCycle3 (gated write-back)
  )
multiCycleTransformerLayer layer kvRamOwner layerIndex processingStateSignal intermediateDataSignal =
  ( nextIntermediateDataSignal
  , writeDoneThisLayerSignal
  , attentionDoneThisLayerSignal
  , commitCycle3Signal
  )
 where
  mha  = multiHeadAttention layer
  ffn  = feedforwardNetwork layer

  -- Drive all KV banks; collect per-head outputs, head-done pulses, and per-bank write-done
  (perHeadOutputSignalsVec, perHeadDoneSignalsVec, perBankWriteDoneVec) =
    let initHeadOutputs = repeat (pure (repeat 0))
        initHeadDone    = repeat (pure False)
        initWriteDone   = repeat (pure False)
    in  foldl
          (fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
          (initHeadOutputs, initHeadDone, initWriteDone)
          indicesI

  -- Attention done: rising edge once all heads finish for this layer
  allHeadsDoneSignal     = fmap and (sequenceA perHeadDoneSignalsVec)
  allHeadsDonePrevSignal = register False allHeadsDoneSignal
  attentionDoneThisLayerSignal =
    liftA2 (\now prev -> now && not prev) allHeadsDoneSignal allHeadsDonePrevSignal

  -- Default per-stage work within this layer
  baseNextIntermediateDataSignal =
    liftA2 (processStage mha ffn layerIndex) processingStateSignal intermediateDataSignal

  -- Layer write-done = AND across banks (Stage2_WriteKV)
  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap and (sequenceA perBankWriteDoneVec)
    in  (\ps banksDone ->
           processingStage ps == Stage2_WriteKV
        && processingLayer ps == layerIndex
        && banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  -- Per-head WO @ head, then sum across heads (equivalent to WO @ concatHeads)
  perHeadProjectedSignalsVec =
    zipWith (\wo hSig -> matrixVectorMultF wo <$> hSig) (MultiHeadAttention.mWo mha) perHeadOutputSignalsVec

  perHeadProjectedSignal = sequenceA perHeadProjectedSignalsVec
  woHeadsSignal          = fmap (foldl1 (zipWith (+))) perHeadProjectedSignal

  -- x_after_attn = x + WO@heads
  xAfterAttnSignal =
      liftA2
        (\idata woHeads ->
          let xInput = inputVector idata
              summed = zipWith (+) xInput woHeads
          in summed)
        intermediateDataSignal
        woHeadsSignal

  -- Commit attention output on this layer’s attnDone pulse in Stage3_Attend.
  -- Print the exact vector being committed (first 8 elems) once per (L, P).
  nextIntermediateDataSignal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal baseNextIntermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

  -- The same gated-commit view, exposed as a tap at Cycle3 (no trace here to avoid duplicate prints)
  commitCycle3Signal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal intermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

processStage
  :: MultiHeadAttention.MultiHeadAttentionComponent
  -> FeedForwardNetwork.FeedForwardNetworkComponent
  -> Index NumLayers
  -> ProcessingState
  -> IntermediateData
  -> IntermediateData
processStage mha ffn layerIndex ps idata
  | processingLayer ps /= layerIndex = idata
  | otherwise = case processingStage ps of

      -- Stage1: compute Q,K,V for current layer/pos
      Stage1_ProjectQKV ->
        let
          (qs, ks, vs) = MultiHeadAttention.projectQKV mha (sequencePosition ps) (inputVector idata)
        in idata { queryVectors = qs, keyVectors = ks, valueVectors = vs }

      -- Stage2: write K,V(pos) to cache
      Stage2_WriteKV -> idata

      -- Stage3: stream attention (sequenced outside).
      Stage3_Attend -> idata

      -- Stage4: FFN
      Stage4_FeedForward ->
        let
          ffnOut = FeedForwardNetwork.computeFeedForward ffn (attentionOutput idata)
        in idata { feedForwardOutput = ffnOut }

      -- Stage5: bookkeeping only
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

-- Access per-head vectors from IntermediateData
getQueryVector :: Signal dom IntermediateData -> Index NumQueryHeads -> Signal dom (Vec HeadDimension FixedPoint)
getQueryVector idSig qIx = (\i -> queryVectors i !! qIx) <$> idSig

getKeyVector :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension FixedPoint)
getKeyVector idSig kvIx = (\i -> keyVectors i !! kvIx) <$> idSig

getValueVector :: Signal dom IntermediateData -> Index NumKeyValueHeads -> Signal dom (Vec HeadDimension FixedPoint)
getValueVector idSig kvIx = (\i -> valueVectors i !! kvIx) <$> idSig

fillOneBank :: HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Cache.KVRamOwner dom
  -> Signal dom IntermediateData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
fillOneBank layerIx psSig kvOwner idSig (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  let
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIx)
             psSig (pure ())

    isStage3Attention = stageEquals Stage3_Attend
    isStage2Write     = stageEquals Stage2_WriteKV

    seqPosSignal = sequencePosition <$> psSig

    bank   = Cache.kvBanks kvOwner !! kvIx
    runKm  = Cache.runKeyMantBank   bank
    runKe  = Cache.runKeyExpBank    bank
    runVm  = Cache.runValueMantBank bank
    runVe  = Cache.runValueExpBank  bank

    qIdx0 = queryHeadIndex0 kvIx
    hasQ1 = hasSecondQueryHead kvIx
    qIdx1 = queryHeadIndex1 kvIx

    query0 = getQueryVector idSig qIdx0
    query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)

    keyVec   = getKeyVector   idSig kvIx
    valueVec = getValueVector idSig kvIx

    keyValuePairSignal = liftA2 (,) keyVec valueVec

    -- Quantize and generate writes
    (writeAddrSig, kMantWr, kExpWr, vMantWr, vExpWr, writeDoneThisBank) =
      Cache.writeSequencer isStage2Write seqPosSignal keyValuePairSignal

    -- Wire BRAMs (Parked reads; Port B used for writes in Stage2)
    addrMantA = pure 0
    addrMantB = mux isStage2Write writeAddrSig (pure 0)

    wrMantA = pure Nothing
    wrMantB_K = mux isStage2Write kMantWr (pure Nothing)
    wrMantB_V = mux isStage2Write vMantWr (pure Nothing)

    addrExpA  = pure 0
    -- Combine kExpWr and vExpWr, prioritizing kExpWr if both are Just
    addrExpB = mux isStage2Write
                   (fmap (maybe 0 fst) (liftA2 chooseWrite kExpWr vExpWr))
                   (pure 0)

    wrExpA    = pure Nothing
    wrExpB_K  = mux isStage2Write kExpWr (pure Nothing)
    wrExpB_V  = mux isStage2Write vExpWr (pure Nothing)

    (_kMantRA, _kMantRB) = runKm (addrMantA, wrMantA) (addrMantB, wrMantB_K)
    (_vMantRA, _vMantRB) = runVm (addrMantA, wrMantA) (addrMantB, wrMantB_V)
    (_kExpRA,  _kExpRB ) = runKe (addrExpA,  wrExpA ) (addrExpB,  wrExpB_K)
    (_vExpRA,  _vExpRB ) = runVe (addrExpA,  wrExpA ) (addrExpB,  wrExpB_V)

    -- Existing register mirror: unchanged
    kvKeysAll = mealy
      (\mem (we, p, rowK) ->
         let mem' = if we then replace p rowK mem else mem
         in  (mem', mem'))
      (repeat (repeat 0))
      (bundle (isStage2Write, seqPosSignal, keyVec))

    kvValsAll = mealy
      (\mem (we, p, rowV) ->
         let mem' = if we then replace p rowV mem else mem
         in  (mem', mem'))
      (repeat (repeat 0))
      (bundle (isStage2Write, seqPosSignal, valueVec))

    out0 = liftA4 attendHeadF
                     query0 kvKeysAll kvValsAll seqPosSignal
    out1raw = liftA4 attendHeadF
                       query1 kvKeysAll kvValsAll seqPosSignal
    out1 = if hasQ1 then out1raw else pure (repeat 0)

    attnPrev = register False isStage3Attention
    donePulse = liftA2 (\now prev -> now && not prev) isStage3Attention attnPrev

    headOutAcc0  = replace qIdx0 out0 headOutAcc
    headDoneAcc0 = replace qIdx0 donePulse headDoneAcc
    headOutAcc1  = if hasQ1 then replace qIdx1 out1 headOutAcc0 else headOutAcc0
    headDoneAcc1 = if hasQ1 then replace qIdx1 donePulse headDoneAcc0 else headDoneAcc0

    writeDoneAcc1 = replace kvIx writeDoneThisBank writeDoneAcc

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1)

-- Helper function to combine kExpWr and vExpWr
chooseWrite
  :: Maybe (Index SequenceLength, ExpS)
  -> Maybe (Index SequenceLength, ExpS)
  -> Maybe (Index SequenceLength, ExpS)
chooseWrite kExp vExp = case kExp of
  Just _  -> kExp  -- Prioritize kExpWr if it has a value
  Nothing -> vExp  -- Fall back to vExpWr
