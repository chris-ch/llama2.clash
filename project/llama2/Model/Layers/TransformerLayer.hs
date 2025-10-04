module Model.Layers.TransformerLayer (
    multiCycleTransformerLayer
    , getKeyVector
    , getValueVector
    , getQueryVector
    , queryHeadIndex1
    , queryHeadIndex0
    , hasSecondQueryHead
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
import Model.Helpers.MatVecI8E (sequentialMatVec, sequentialMatVecStub)
import Model.Numeric.Types (FixedPoint)
import Model.Helpers (liftA4)
import Model.Layers.Attention.AttendSequential (attendHeadSeq)
import Model.Memory.RamOps (toRamOperation)
import Model.Numeric.ParamPack (QArray2D)

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
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom LayerData
  -> ( Signal dom LayerData
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom LayerData
  )
multiCycleTransformerLayer layer layerIndex processingState layerData =
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
          (fillOneBank layerIndex processingState layerData)
          (initHeadOutputs, initHeadDone, initWriteDone)
          indicesI

  baseNextLayerData :: Signal dom LayerData
  baseNextLayerData =
    liftA2 (stageProcessor mha ffn layerIndex) processingState layerData

  allBanksDone :: Signal dom Bool
  allBanksDone = and <$> sequenceA perBankWriteDoneFlags

  writeDone :: Signal dom Bool
  writeDone = kvWriteDoneCond layerIndex <$> processingState <*> allBanksDone

  -- === Per-head WO projection (with proper handshaking) ===
  (perHeadProjected, allWODone) = 
    perHeadWOController perHeadOutputs perHeadDoneFlags (mWoQ mha)

  -- Combine projected heads into WO result (only valid when allWODone)
  -- First convert Vec of Signals to Signal of Vec, then sum
  woHeads :: Signal dom (Vec ModelDimension FixedPoint)
  woHeads = foldl1 (zipWith (+)) <$> sequenceA perHeadProjected

  -- WO projection is complete when all heads are projected
  validProjected :: Signal dom Bool
  validProjected = allWODone

  xAfterAttn :: Signal dom (Vec ModelDimension FixedPoint)
  xAfterAttn = liftA2 inputsAggregator layerData woHeads

  -- Attention done pulses on rising edge of validProjected
  attentionDone :: Signal dom Bool
  attentionDone = 
    let prevValid = register False validProjected
        risingEdge = validProjected .&&. (not <$> prevValid)
    in risingEdge

  nextLayerData :: Signal dom LayerData
  nextLayerData = liftA4 (layerDataAttnDone layerIndex) processingState baseNextLayerData xAfterAttn attentionDone

  commitStage3 :: Signal dom LayerData
  commitStage3 = liftA4 (layerDataAttnDone layerIndex)  processingState layerData xAfterAttn attentionDone

-- Per-head WO projection with proper handshaking
perHeadWOController ::
  forall dom .
  HiddenClockResetEnable dom
  => Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))  -- head outputs
  -> Vec NumQueryHeads (Signal dom Bool)                            -- head done flags
  -> Vec NumQueryHeads (QArray2D ModelDimension HeadDimension)      -- WO matrices
  -> ( Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint))  -- projected outputs
     , Signal dom Bool                                                  -- all projections done
     )
perHeadWOController perHeadOutputs perHeadDoneFlags mWoQs =
  (perHeadProjected, allProjectionsDone)
  where
    -- For each head: create a controller that starts WO projection when head completes
    perHeadResults = 
      zipWith3 controlOneHead perHeadOutputs perHeadDoneFlags mWoQs
    
    -- Unpack results
    perHeadProjected = map (\(out, _, _) -> out) perHeadResults
    perHeadValidOuts = map (\(_, v, _) -> v) perHeadResults
    
    -- All projections done when all validOut signals are high
    allProjectionsDone = and <$> sequenceA perHeadValidOuts

-- Controller for one head's WO projection
controlOneHead ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Vec HeadDimension FixedPoint)           -- head output
  -> Signal dom Bool                                      -- head done
  -> QArray2D ModelDimension HeadDimension               -- WO matrix
  -> ( Signal dom (Vec ModelDimension FixedPoint)        -- projected output
     , Signal dom Bool                                    -- validOut
     , Signal dom Bool                                    -- readyOut
     )
controlOneHead headOutput headDone woMatrix = (projOut, validOut, readyOut)
  where
    -- Detect rising edge of headDone
    headDonePrev = register False headDone
    headDoneRising = headDone .&&. (not <$> headDonePrev)
    
    -- State: IDLE (0) -> PROJECTING (1) -> DONE (2)
    state :: Signal dom (Unsigned 2)
    state = register 0 nextState
    
    nextState = 
      mux (state .==. 0 .&&. headDoneRising) (pure 1) $  -- Rising edge of headDone -> start
      mux (state .==. 1 .&&. woValidOut) (pure 2) $      -- WO done -> done state
      mux (state .==. 2) (pure 0) $                      -- Reset to idle next cycle
      state                                              -- Hold state
    
    -- Start WO projection when entering state 1
    startWO = state .==. 0 .&&. headDoneRising
    
    -- Call the sequential matmul stub
    (woResult, woValidOut, woReadyOut) = 
      sequentialMatVecStub woMatrix (bundle (startWO, headOutput))
    
    -- Output the result (hold it when valid)
    projOut = regEn (repeat 0) woValidOut woResult
    
    -- Valid out in DONE state (state 2)
    validOut = state .==. 2
    
    -- Ready when idle
    readyOut = state .==. 0

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
  -> Signal dom LayerData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
fillOneBank layerIndex psSig idSig (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  let
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIndex)
             psSig (pure ())

    isStage3Attention = stageEquals Stage3_Attend
    isStage2Write     = stageEquals Stage2_WriteKV

    attnPrev = register False isStage3Attention
    clearS3  = liftA2 (\now prev -> now && not prev) isStage3Attention attnPrev

    seqPosSignal = sequencePosition <$> psSig

    qIdx0 = queryHeadIndex0 kvIx
    hasQ1 = hasSecondQueryHead kvIx
    qIdx1 = queryHeadIndex1 kvIx

    query0 = getQueryVector idSig qIdx0
    query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)

    keyVec   = getKeyVector   idSig kvIx
    valueVec = getValueVector idSig kvIx

    -- Generate a single write pulse per Stage2, and use it for both K and V.
    (wrPulse, wrDonePulse) = Cache.writeOnce isStage2Write

    wrKVRowK = mux wrPulse (Just <$> bundle (seqPosSignal, keyVec))   (pure Nothing)
    wrKVRowV = mux wrPulse (Just <$> bundle (seqPosSignal, valueVec)) (pure Nothing)

    (kRowA, _kRowB) =
      trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                           (toRamOperation seqPosSignal wrKVRowK)

    (vRowA, _vRowB) =
      trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                           (toRamOperation seqPosSignal wrKVRowV)

    (tAddrRow, stepEnRow, lastTRow) =
      attentionRowSequencer clearS3 isStage3Attention seqPosSignal

    (out0_seqF, done0_seqF) = attendHeadSeq clearS3 stepEnRow query0 kRowA vRowA lastTRow
    (out1_seqF, done1_seqF) =
      if hasQ1 then attendHeadSeq clearS3 stepEnRow query1 kRowA vRowA lastTRow
               else (pure (repeat 0), pure False)

    headOutAcc0  = replace qIdx0 out0_seqF headOutAcc
    headOutAcc1  = if hasQ1 then replace qIdx1 out1_seqF headOutAcc0 else headOutAcc0

    headDoneAcc0 = replace qIdx0 done0_seqF headDoneAcc
    headDoneAcc1 = if hasQ1 then replace qIdx1 done1_seqF headDoneAcc0 else headDoneAcc0

    writeDoneAcc1 = replace kvIx wrDonePulse writeDoneAcc

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1)

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
