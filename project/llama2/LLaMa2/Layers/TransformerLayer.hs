module LLaMa2.Layers.TransformerLayer (
    transformerLayer
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
import LLaMa2.Layers.TransformerLayer.Internal
import LLaMa2.Core.Types
  ( ProcessingState(..), LayerData(..), CycleStage(..)
  )
import LLaMa2.Config
  ( ModelDimension
  , NumLayers, NumQueryHeads, NumKeyValueHeads
  , HeadDimension,  SequenceLength
  )
import qualified LLaMa2.Memory.KVCacheBank as Cache

import LLaMa2.Layers.Components.Quantized
  ( FeedForwardNetworkComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , EmbeddingComponentQ(..)
  )

import qualified LLaMa2.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork (feedForwardStage)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layers.Attention.AttentionHead (attendHead)
import LLaMa2.Memory.RamOps (runTdpRam)
import LLaMa2.Numeric.ParamPack (MatI8E)
import LLaMa2.Layers.Attention.MultiHeadAttention (projectQKV)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttentionComponentQ
  , feedforwardNetwork :: FeedForwardNetworkComponentQ
  } deriving (Show)

data TransformerDecoderComponent = TransformerDecoderComponent
  { modelEmbedding :: EmbeddingComponentQ
  , modelLayers    :: Vec NumLayers TransformerLayerComponent
  } deriving (Show)

data LayerFlowState = LIdle | LQKV | LWrite | LAttn | LFFN | LDone
  deriving (Show, Eq, Generic, NFDataX)

layerFlowFsm ::
  HiddenClockResetEnable dom =>
  Signal dom Bool ->  -- validIn
  Signal dom Bool ->  -- readyOut
  ( Signal dom LayerFlowState
  , Signal dom Bool        -- validOut
  , Signal dom Bool        -- readyIn
  )
layerFlowFsm validIn readyOut =
  mealyB go LIdle (validIn, readyOut)
 where
  go s (vIn, rOut) =
    case s of
      LIdle | vIn     -> (LQKV,  (LQKV,  False, True))
      LQKV             -> (LWrite,(LWrite,False, True))
      LWrite           -> (LAttn, (LAttn, False, True))
      LAttn            -> (LFFN,  (LFFN,  False, True))
      LFFN             -> (LDone, (LDone, True,  rOut))
      LDone | rOut    -> (LIdle,  (LIdle, False, True))
            | otherwise -> (LDone, (LDone, True,  False))

transformerLayer :: forall dom . HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom LayerData
  -> ( Signal dom LayerData
     , Signal dom Bool           -- writeDone
     , Signal dom Bool           -- attentionDone
     , Signal dom Bool           -- qkvDone
     , Signal dom LayerData      -- layerDataAfterAttention
     , Signal dom Bool           -- qkvInReady
     , Signal dom Bool           -- ffnDone
     )
transformerLayer layer layerIndex processingState layerData =
  ( nextLayerData
  , writeDone
  , attentionDone
  , qkvDone
  , layerDataAfterAttention
  , qkvInReady
  , ffnValidOut
  )
 where
  mha = multiHeadAttention layer
  ffn = feedforwardNetwork layer

  -- === Stage 1: QKV Projection ===
  isStage1ThisLayer :: Signal dom Bool
  isStage1ThisLayer = (\ps -> processingStage ps == Stage1_ProjectQKV
                           && processingLayer ps == layerIndex)
                      <$> processingState

  -- QKV outputs only when next stage ready
  qkvOutReady :: Signal dom Bool
  qkvOutReady = (\ps -> processingStage ps == Stage2_WriteKV
                     && processingLayer ps == layerIndex)
                <$> processingState

  ( qkvProjected, qkvValidOut, qkvInReady ) =
    qkvProjectionController
      isStage1ThisLayer
      qkvOutReady
      layerData
      mha
      processingState

  qkvDone = qkvValidOut

  -- === Stage 2/3: KV Cache Write & Attention ===
  ( perHeadOutputs, perHeadDoneFlags, perBankWriteDoneFlags ) =
    foldl
      ( fillOneBank layerIndex processingState layerData qkvDone )
      ( repeat (pure (repeat 0))
      , repeat (pure False)
      , repeat (pure False)
      )
      indicesI

  baseNextLayerData =
    updateLayerDataForStage layerIndex
      <$> processingState <*> layerData <*> qkvProjected

  allBanksDone = and <$> sequenceA perBankWriteDoneFlags
  writeDone    = kvWriteDoneCond layerIndex <$> processingState <*> allBanksDone

  -- === Stage 3b: WO Projection ===
  ( perHeadProjected
    , perHeadValidOuts
    , perHeadReadyOuts
    ) =
      perHeadWOController perHeadOutputs perHeadDoneFlags (mWoQ mha)

  gatedHeads =
    zipWith3
      (\proj vld rdy -> mux rdy proj (pure (repeat 0)))
      perHeadProjected perHeadValidOuts perHeadReadyOuts

  woHeads = foldl1 (zipWith (+)) <$> sequenceA gatedHeads
  validProjected = and <$> sequenceA perHeadReadyOuts

  xAfterAttn = inputsAggregator <$> layerData <*> woHeads

  attentionDone =
    let prevReady = register False validProjected
    in validProjected .&&. (not <$> prevReady)

  -- === Stage 4: Feed-Forward ===
  isStage4ThisLayer :: Signal dom Bool
  isStage4ThisLayer = (\ps -> processingStage ps == Stage4_FeedForward
                           && processingLayer ps == layerIndex)
                      <$> processingState

  ffnInput = attentionOutput <$> layerDataAfterAttention

  ffnOutReady =
    (\ps -> case () of
       _ | processingStage ps == Stage1_ProjectQKV
         && processingLayer ps == layerIndex + 1 -> True
         | processingStage ps == Stage5_Classifier
         && processingLayer ps == maxBound -> True
         | otherwise -> False)
    <$> processingState

  ( ffnOutput, ffnValidOut, _ffnInReady ) =
    ffnController
      isStage4ThisLayer
      ffnOutReady
      ffnInput
      ffn

  numLayer = pure layerIndex

  nextLayerData =
    layerDataWithFFN <$> numLayer
      <*> processingState
      <*> baseNextLayerData
      <*> xAfterAttn
      <*> attentionDone
      <*> ffnOutput
      <*> ffnValidOut

  layerDataAfterAttention =
    layerDataAttnDone <$> numLayer
      <*> processingState
      <*> layerData
      <*> xAfterAttn
      <*> attentionDone

layerDataWithFFN :: Index NumLayers
  -> ProcessingState
  -> LayerData
  -> Vec ModelDimension FixedPoint
  -> Bool
  -> Vec ModelDimension FixedPoint
  -> Bool
  -> LayerData
layerDataWithFFN layerIndex ps baseData attnOut attnDone ffnOut ffnValid =
  let withAttn = layerDataAttnDone layerIndex ps baseData attnOut attnDone
  in if processingLayer ps == layerIndex
        && processingStage ps == Stage4_FeedForward
        && ffnValid
        then withAttn { feedForwardOutput = ffnOut }
        else withAttn

data FFNState = FFNIdle | FFNComputing | FFNDone
  deriving (Show, Eq, Generic, NFDataX)

ffnController ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool                              -- inValid (FFN input valid)
  -> Signal dom Bool                              -- outReady (downstream ready)
  -> Signal dom (Vec ModelDimension FixedPoint)   -- input vector
  -> FeedForwardNetworkComponentQ                 -- FFN quantized params
  -> ( Signal dom (Vec ModelDimension FixedPoint) -- result
     , Signal dom Bool                            -- validOut
     , Signal dom Bool                            -- inReady
     )
ffnController inValid outReady inputVec ffnQ = (result, outValid, inReady)
  where
    -- === FSM ===
    state :: Signal dom FFNState
    state = register FFNIdle nextState

    inReady  = state .==. pure FFNIdle
    startTx  = inValid .&&. inReady
    outValid = state .==. pure FFNDone
    consume  = outValid .&&. outReady

    nextState =
      mux (state .==. pure FFNIdle)
          (mux startTx (pure FFNComputing) (pure FFNIdle))
          (mux (state .==. pure FFNComputing)
              (mux ffnSeqValid (pure FFNDone) (pure FFNComputing))
              (mux consume (pure FFNIdle) (pure FFNDone)))

    -- === Compute enable + internal ready ===
    computeEnable :: Signal dom Bool
    computeEnable = startTx .||. (state .==. pure FFNComputing)

    -- Allow FFN internal pipeline to run only when computing or consumer ready
    internalReady :: Signal dom Bool
    internalReady = mux (state .==. pure FFNComputing)
                        (pure True)     -- during compute, internal pipeline free-runs
                        outReady        -- when done, wait until consumer ready

    -- === Call FFN core ===
    (result, ffnSeqValid, _ffnSeqReady) =
      FeedForwardNetwork.feedForwardStage
        computeEnable
        internalReady   -- backpressure-aware signal
        ffnQ
        inputVec

data QKVState = QKVIdle | QKVComputing | QKVDone
  deriving (Show, Eq, Generic, NFDataX)

qkvProjectionController ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom LayerData
  -> MultiHeadAttentionComponentQ
  -> Signal dom ProcessingState
  -> ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     )
qkvProjectionController inValid outReady idSig mhaQ psSig = (result, outValid, inReady)
  where
    state :: Signal dom QKVState
    state = register QKVIdle nextState

    inReady :: Signal dom Bool
    inReady = state .==. pure QKVIdle

    startTx :: Signal dom Bool
    startTx = inValid .&&. inReady

    outValid :: Signal dom Bool
    outValid = state .==. pure QKVDone

    consume :: Signal dom Bool
    consume = outValid .&&. outReady

    nextState :: Signal dom QKVState
    nextState =
      mux (state .==. pure QKVIdle)
          (mux startTx (pure QKVComputing) (pure QKVIdle))
          (mux (state .==. pure QKVComputing)
              (mux matVecValid (pure QKVDone) (pure QKVComputing))
              (mux consume (pure QKVIdle) (pure QKVDone)))

    computeEnable = startTx .||. (state .==. pure QKVComputing)

    -- Internal ready should propagate actual downstream state
    -- When done (QKVDone), wait for outReady before accepting new input
    -- When computing, allow pipeline to proceed
    internalReady = mux (state .==. pure QKVComputing)
                        (pure True)     -- Free-run during computation
                        outReady        -- Respect backpressure when done
    
    (result, matVecValid, _matVecReady) =
      projectQKV
        computeEnable
        internalReady   -- propagate backpressure
        mhaQ
        (sequencePosition <$> psSig)
        (inputVector <$> idSig)

perHeadWOController ::
  forall dom .
  HiddenClockResetEnable dom
  => Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
  -> Vec NumQueryHeads (Signal dom Bool)
  -> Vec NumQueryHeads (MatI8E ModelDimension HeadDimension)
  -> ( Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumQueryHeads (Signal dom Bool)
     )
perHeadWOController perHeadOutputs perHeadDoneFlags mWoQs =
  (perHeadProjected, perHeadValidOuts, perHeadReadyOuts)
  where
    headValidIn = zipWith (.&&.) perHeadDoneFlags perHeadReadyOuts
    
    perHeadResults = zipWith3 singleHeadController headValidIn perHeadOutputs mWoQs

    perHeadProjected = map (\(result, _, _) -> result) perHeadResults
    perHeadValidOuts = map (\(_, validOut, _) -> validOut) perHeadResults
    perHeadReadyOuts = map (\(_, _, readyOut) -> readyOut) perHeadResults

-- Helper functions
kvWriteDoneCond :: Index NumLayers -> ProcessingState -> Bool -> Bool
kvWriteDoneCond layerIndex state banksDone = 
  processingStage state == Stage2_WriteKV
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

updateLayerDataForStage ::
  Index NumLayers
  -> ProcessingState
  -> LayerData
  -> (Vec NumQueryHeads (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
     , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  -> LayerData
updateLayerDataForStage layerIndex ps idata (qs, ks, vs)
  | processingLayer ps /= layerIndex = idata
  | otherwise = case processingStage ps of
      Stage1_ProjectQKV ->
        idata { queryVectors = qs, keyVectors = ks, valueVectors = vs }
      _ -> idata

queryHeadsPerKeyValueHead :: Int
queryHeadsPerKeyValueHead = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

maxQueryHeadIndex :: Int
maxQueryHeadIndex = natToNum @NumQueryHeads - 1

baseQueryIndex :: Index NumKeyValueHeads -> Int
baseQueryIndex kvIx = fromEnum kvIx * queryHeadsPerKeyValueHead

queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 kvIx = toEnum (min maxQueryHeadIndex (baseQueryIndex kvIx))

hasSecondQueryHead :: Index NumKeyValueHeads -> Bool
hasSecondQueryHead kvIx = 
  queryHeadsPerKeyValueHead >= 2 && (baseQueryIndex kvIx + 1 <= maxQueryHeadIndex)

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

fillOneBank ::
  forall dom .
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom LayerData
  -> Signal dom Bool  -- FIX: Add qkvValid parameter
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool) )
fillOneBank layerIndex psSig idSig qkvValid (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  let
    stageEquals st =
      liftA2 (\ps _ -> processingStage ps == st && processingLayer ps == layerIndex)
             psSig (pure ())

    isStage3Attention = stageEquals Stage3_Attend
    isStage2Write     = stageEquals Stage2_WriteKV

    attnPrev = register False isStage3Attention
    risingEdge now prev = now && not prev
    clearS3  = risingEdge <$> isStage3Attention <*> attnPrev

    seqPosSignal = sequencePosition <$> psSig

    qIdx0 = queryHeadIndex0 kvIx
    hasQ1 = hasSecondQueryHead kvIx
    qIdx1 = queryHeadIndex1 kvIx

    query0 = getQueryVector idSig qIdx0
    query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)

    keyVec   = getKeyVector   idSig kvIx
    valueVec = getValueVector idSig kvIx

    -- FIX: Gate write pulse with qkvValid to ensure data is ready
    (wrPulse, wrDonePulse) = Cache.writeOnce (isStage2Write .&&. qkvValid)

    wrKVRowK = mux wrPulse (Just <$> bundle (seqPosSignal, keyVec))   (pure Nothing)
    wrKVRowV = mux wrPulse (Just <$> bundle (seqPosSignal, valueVec)) (pure Nothing)

    (kRowA, _kRowB) =
      runTdpRam tAddrRow
                (pure Nothing)
                seqPosSignal
                wrKVRowK

    (vRowA, _vRowB) =
      runTdpRam tAddrRow
                (pure Nothing)
                seqPosSignal
                wrKVRowV

    (tAddrRow, stepEnRow, lastTRow) =
      attentionRowSequencer clearS3 isStage3Attention seqPosSignal

    (out0_seqF, done0_seqF) = attendHead clearS3 stepEnRow query0 kRowA vRowA lastTRow
    (out1_seqF, done1_seqF) =
      if hasQ1 then attendHead clearS3 stepEnRow query1 kRowA vRowA lastTRow
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
