module Model.Layers.TransformerLayer (
    multiCycleTransformerLayer
  , TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
) where

import Clash.Prelude
import Model.Core.Types
  ( ModelDimemsion
  , NumLayers, NumQueryHeads, NumKeyValueHeads
  , HeadDimension, ProcessingState(..), IntermediateData(..), CycleStage(..), SequenceLength
  )
import qualified Model.Memory.KVCacheBank as Cache

import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention (projectQKV)
import Model.Layers.Components.Quantized
  ( FeedForwardNetworkComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , EmbeddingComponentQ(..)
  )

import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork (computeFeedForward)
import Model.Helpers.MatVecI8E (matrixVectorMultI8E_Fixed)
import Model.Numeric.Types (ExpS, FixedPoint)
import Helpers (liftA4)
import Model.Layers.Attention.AttentionHead.Fixed (attendHeadF)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttentionComponentQ
  , feedforwardNetwork :: FeedForwardNetworkComponentQ
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponentQ
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
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom IntermediateData
  )
multiCycleTransformerLayer layer kvRamOwner layerIndex processingStateSignal intermediateDataSignal =
  ( nextIntermediateDataSignal
  , writeDoneThisLayerSignal
  , attentionDoneThisLayerSignal
  , commitCycle3Signal
  )
 where
  mhaQ = multiHeadAttention layer
  ffnQ = feedforwardNetwork layer

  (perHeadOutputSignalsVec, perHeadDoneSignalsVec, perBankWriteDoneVec) =
    let initHeadOutputs = repeat (pure (repeat 0))
        initHeadDone    = repeat (pure False)
        initWriteDone   = repeat (pure False)
    in  foldl
          (fillOneBank layerIndex processingStateSignal kvRamOwner intermediateDataSignal)
          (initHeadOutputs, initHeadDone, initWriteDone)
          indicesI

  allHeadsDoneSignal     = fmap and (sequenceA perHeadDoneSignalsVec)
  allHeadsDonePrevSignal = register False allHeadsDoneSignal
  attentionDoneThisLayerSignal =
    liftA2 (\now prev -> now && not prev) allHeadsDoneSignal allHeadsDonePrevSignal

  baseNextIntermediateDataSignal =
    liftA2 (processStage mhaQ ffnQ layerIndex) processingStateSignal intermediateDataSignal

  writeDoneThisLayerSignal =
    let allBanksDoneSignal = fmap and (sequenceA perBankWriteDoneVec)
    in  (\ps banksDone ->
           processingStage ps == Stage2_WriteKV
        && processingLayer ps == layerIndex
        && banksDone)
        <$> processingStateSignal <*> allBanksDoneSignal

  -- Per-head WO projection now uses quantized WO blocks (I8E) -> FixedPoint
  perHeadProjectedSignalsVec =
    zipWith (\woQ hSig -> matrixVectorMultI8E_Fixed woQ <$> hSig) (mWoQ mhaQ) perHeadOutputSignalsVec

  perHeadProjectedSignal = sequenceA perHeadProjectedSignalsVec
  woHeadsSignal          = fmap (foldl1 (zipWith (+))) perHeadProjectedSignal

  xAfterAttnSignal =
      liftA2
        (\idata woHeads ->
          let xInput = inputVector idata
          in zipWith (+) xInput woHeads)
        intermediateDataSignal
        woHeadsSignal

  nextIntermediateDataSignal =
    liftA4
      (\ps cur attOut done ->
         if processingLayer ps == layerIndex
            && processingStage ps == Stage3_Attend
            && done
           then cur { attentionOutput = attOut }
           else cur)
      processingStateSignal baseNextIntermediateDataSignal xAfterAttnSignal attentionDoneThisLayerSignal

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
  :: MultiHeadAttentionComponentQ
  -> FeedForwardNetworkComponentQ
  -> Index NumLayers
  -> ProcessingState
  -> IntermediateData
  -> IntermediateData
processStage mhaQ ffnQ layerIndex ps idata
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

    (writeAddrSig, kMantWr, kExpWr, vMantWr, vExpWr, writeDoneThisBank) =
      Cache.writeSequencer isStage2Write seqPosSignal keyValuePairSignal

    addrMantA = pure 0
    addrMantB = mux isStage2Write writeAddrSig (pure 0)

    wrMantA = pure Nothing
    wrMantB_K = mux isStage2Write kMantWr (pure Nothing)
    wrMantB_V = mux isStage2Write vMantWr (pure Nothing)

    addrExpA  = pure 0
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

chooseWrite
  :: Maybe (Index SequenceLength, ExpS)
  -> Maybe (Index SequenceLength, ExpS)
  -> Maybe (Index SequenceLength, ExpS)
chooseWrite kExp vExp = case kExp of
  Just _  -> kExp
  Nothing -> vExp
