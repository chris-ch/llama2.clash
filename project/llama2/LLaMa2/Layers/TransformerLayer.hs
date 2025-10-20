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
import LLaMa2.Layers.Attention.AttentionHead (attentionHead)
import LLaMa2.Memory.RamOps (trueDualPortRam)
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

-- Shared 3-state valid/ready controller
data GenericState = Idle | Compute | Done
  deriving (Show, Eq, Generic, NFDataX)

-- FSM states for autonomous stages
data WriteState = WriteIdle | WriteWriting | WriteDone
  deriving (Show, Eq, Generic, NFDataX)

data AttnState = AttnIdle | AttnComputing | AttnDone
  deriving (Show, Eq, Generic, NFDataX)

fsmController ::
  HiddenClockResetEnable dom =>
  Signal dom Bool ->  -- inValid
  Signal dom Bool ->  -- outReady
  Signal dom Bool ->  -- computeDone
  ( Signal dom Bool   -- enable
  , Signal dom Bool   -- validOut
  , Signal dom Bool   -- inReady
  )
fsmController inValid outReady computeDone = (enable, validOut, inReady)
 where
  state    = register Idle next
  inReady  = state .==. pure Idle
  startTx  = inValid .&&. inReady
  validOut = state .==. pure Done
  consume  = validOut .&&. outReady

  next = mux (state .==. pure Idle)
              (mux startTx (pure Compute) (pure Idle))
              (mux (state .==. pure Compute)
                  (mux computeDone (pure Done) (pure Compute))
                  (mux consume (pure Idle) (pure Done)))

  enable = startTx .||. (state .==. pure Compute)

kvWriteControllerFSM ::
  HiddenClockResetEnable dom =>
  Signal dom Bool ->  -- validIn (QKV done)
  Signal dom Bool ->  -- readyOut (Attn ready)
  Signal dom Bool ->  -- writeComplete (all banks written)
  ( Signal dom Bool   -- validOut (write done, ready for attn)
  , Signal dom Bool   -- readyIn (can accept QKV)
  , Signal dom Bool   -- enableWrite (trigger write)
  )
kvWriteControllerFSM validIn readyOut writeComplete = (validOut, readyIn, enableWrite)
 where
  state = register WriteIdle nextState
  
  readyIn = state .==. pure WriteIdle
  startWrite = validIn .&&. readyIn
  validOut = state .==. pure WriteDone
  consume = validOut .&&. readyOut
  
  nextState = mux (state .==. pure WriteIdle)
                  (mux startWrite (pure WriteWriting) (pure WriteIdle))
                  (mux (state .==. pure WriteWriting)
                      (mux writeComplete (pure WriteDone) (pure WriteWriting))
                      (mux consume (pure WriteIdle) (pure WriteDone)))
  
  enableWrite = startWrite .||. (state .==. pure WriteWriting)

ffnController ::
  HiddenClockResetEnable dom =>
  Signal dom Bool -> Signal dom Bool ->
  Signal dom (Vec ModelDimension FixedPoint) ->
  FeedForwardNetworkComponentQ ->
  ( Signal dom (Vec ModelDimension FixedPoint)
  , Signal dom Bool
  , Signal dom Bool )
ffnController inValid outReady inputVec ffnQ = (result, validOut, inReady)
 where
  (enable, validOut, inReady) = fsmController inValid outReady ffnSeqValid
  (result, ffnSeqValid, _ready) =
    FeedForwardNetwork.feedForwardStage enable outReady ffnQ inputVec


qkvProjectionController ::
  HiddenClockResetEnable dom =>
  Signal dom Bool -> Signal dom Bool ->
  Signal dom LayerData ->
  MultiHeadAttentionComponentQ ->
  Signal dom ProcessingState ->
  ( Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  , Signal dom Bool
  , Signal dom Bool )
qkvProjectionController inValid outReady idSig mhaQ psSig = (result, validOut, inReady)
 where
  (enable, validOut, inReady) = fsmController inValid outReady matVecValid
  (result, matVecValid, _ready) =
    projectQKV enable (pure True) mhaQ
               (sequencePosition <$> psSig)
               (inputVector <$> idSig)

transformerLayer :: forall dom . HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom Bool -- NEW: layerActive (not used yet)
  -> Signal dom LayerData
  -> ( Signal dom LayerData
     , Signal dom Bool           -- writeDone
     , Signal dom Bool           -- attentionDone
     , Signal dom Bool           -- qkvDone
     , Signal dom LayerData      -- layerDataAfterAttention
     , Signal dom Bool           -- qkvInReady
     , Signal dom Bool           -- ffnDone
     )
transformerLayer layer layerIndex processingState layerActive layerData =
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

  -- === Stage1: QKV Projection ===
  isStage1ThisLayer = (\ps -> processingStage ps == Stage1_ProjectQKV
                               && processingLayer ps == layerIndex) <$> processingState

  qkvOutReady = (\ps -> processingStage ps == Stage2_WriteKV
                         && processingLayer ps == layerIndex) <$> processingState

  (qkvProjected, qkvValidOut, qkvInReady) =
    qkvProjectionController
      isStage1ThisLayer
      qkvOutReady
      layerData
      mha
      processingState

  qkvDone = qkvValidOut

  -- === Stage2/3: KV Cache and Attention ===
  initHeadOutputs = repeat (pure (repeat 0))
  initHeadDone    = repeat (pure False)
  initWriteDone   = repeat (pure False)

  -- Write stage FSM controller - produces writeDone signal
  (writeValidOutNew, _writeReadyIn, _writeEnable) =
    kvWriteControllerFSM 
      qkvValidOut           -- validIn: start when QKV completes
      (pure True)           -- readyOut: always ready (attention doesn't use FSM)
      allBanksDone          -- writeComplete: all banks finished writing

  (perHeadOutputs, perHeadDoneFlags, perBankWriteDoneFlags) =
    foldl
      (fillOneBank layerIndex processingState layerData qkvDone)
      (initHeadOutputs, initHeadDone, initWriteDone)
      indicesI

  baseNextLayerData = updateLayerDataForStage layerIndex <$> processingState <*> layerData <*> qkvProjected

  allBanksDone = and <$> sequenceA perBankWriteDoneFlags
  
  -- Use FSM completion signal instead of stage check
  writeDone = writeValidOutNew

  -- === Per-head WO projection ===
  (perHeadProjected, perHeadValidOuts, perHeadReadyOuts) =
    perHeadWOController perHeadOutputs perHeadDoneFlags (mWoQ mha)

  gatedHeads :: Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint))
  gatedHeads =
    zipWith3 (\proj _valid ready -> mux ready proj (pure (repeat 0)))
             perHeadProjected
             perHeadValidOuts
             perHeadReadyOuts

  woHeads = foldl1 (zipWith (+)) <$> sequenceA gatedHeads
  validProjected = and <$> sequenceA perHeadReadyOuts
  xAfterAttn = inputsAggregator <$> layerData <*> woHeads
  attentionDone = let prevReady = register False validProjected
                  in validProjected .&&. (not <$> prevReady)

  layerDataAfterAttention = (layerDataAttnDone layerIndex <$> processingState)
                                               <*> layerData
                                               <*> xAfterAttn
                                               <*> attentionDone

  -- === Stage4: FFN ===
  isStage4ThisLayer = (\ps -> processingStage ps == Stage4_FeedForward
                                && processingLayer ps == layerIndex) <$> processingState

  ffnInput = attentionOutput <$> layerDataAfterAttention

  ffnOutReady = (\ps -> case () of
                           _ | processingStage ps == Stage1_ProjectQKV
                             && processingLayer ps == layerIndex + 1 -> True
                             | processingStage ps == Stage5_Classifier
                             && processingLayer ps == maxBound -> True
                             | otherwise -> False) <$> processingState

  (ffnOutput, ffnValidOut, ffnInReady) =
    ffnController
      isStage4ThisLayer
      ffnOutReady
      ffnInput
      ffn

  nextLayerData = (layerDataWithFFN layerIndex <$> processingState)
                                   <*> baseNextLayerData
                                   <*> xAfterAttn
                                   <*> attentionDone
                                   <*> ffnOutput
                                   <*> ffnValidOut


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
  forall dom.
  HiddenClockResetEnable dom =>
  Index NumLayers ->
  Signal dom ProcessingState ->
  Signal dom LayerData ->
  Signal dom Bool -> -- qkvValid
  ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
  , Vec NumQueryHeads (Signal dom Bool)
  , Vec NumKeyValueHeads (Signal dom Bool) ) ->
  Index NumKeyValueHeads ->
  ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
  , Vec NumQueryHeads (Signal dom Bool)
  , Vec NumKeyValueHeads (Signal dom Bool) )
fillOneBank layerIndex psSig idSig qkvValid (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
  (headOutAcc2, headDoneAcc2, writeDoneAcc1)
 where
  -- Stage signals
  isStage2Write = liftA2 (\ps _ -> processingStage ps == Stage2_WriteKV &&
                                   processingLayer ps == layerIndex) psSig (pure ())
  isStage3Attn  = liftA2 (\ps _ -> processingStage ps == Stage3_Attend &&
                                   processingLayer ps == layerIndex) psSig (pure ())

  seqPos = sequencePosition <$> psSig

  -- Query indices
  qIdx0 = queryHeadIndex0 kvIx
  hasQ1 = hasSecondQueryHead kvIx
  qIdx1 = queryHeadIndex1 kvIx

  query0 = getQueryVector idSig qIdx0
  query1 = if hasQ1 then getQueryVector idSig qIdx1 else pure (repeat 0)
  keyVec = getKeyVector idSig kvIx
  valVec = getValueVector idSig kvIx

  -- KV Write controller
  (wrPulse, wrDone) = Cache.writeOnce (isStage2Write .&&. qkvValid)
  wrKVRowK = mux wrPulse (Just <$> bundle (seqPos, keyVec)) (pure Nothing)
  wrKVRowV = mux wrPulse (Just <$> bundle (seqPos, valVec)) (pure Nothing)

  (kRow, _kRowB) = trueDualPortRam tAddrRow (pure Nothing) seqPos wrKVRowK
  (vRow, _vRowB) = trueDualPortRam tAddrRow (pure Nothing) seqPos wrKVRowV
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
