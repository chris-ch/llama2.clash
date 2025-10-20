module LLaMa2.Layer.TransformerLayer (
    transformerLayer
    , getKeyVector
    , getValueVector
    , getQueryVector
    , queryHeadIndex1
    , queryHeadIndex0
    , hasSecondQueryHead
    , singleHeadController
  , TransformerLayerComponent(..)
) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig 
    ( ModelDimension,
      HeadDimension,
      NumLayers,
      NumQueryHeads,
      NumKeyValueHeads,
      SequenceLength )
import LLaMa2.Numeric.Operations (matrixMultiplier)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization ( MatI8E, MatI8E )
import LLaMa2.Types.LayerData
  ( ProcessingState(..), LayerData(..), CycleStage(..)
  )
import qualified LLaMa2.Memory.KVCacheBank as Cache

import LLaMa2.Layer.Components.Quantized
  ( FeedForwardNetworkComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  )

import qualified LLaMa2.Layer.FeedForward.FeedForwardNetwork as FeedForwardNetwork (feedForwardStage)
import LLaMa2.Layer.Attention.AttentionHead (attentionHead)
import LLaMa2.Memory.DualPortRAM (trueDualPortRam)
import LLaMa2.Layer.Attention (fsmController)
import LLaMa2.Layer.Attention.QKVProjection (qkvProjectionController)

data TransformerLayerComponent = TransformerLayerComponent
  { multiHeadAttention :: MultiHeadAttentionComponentQ
  , feedforwardNetwork :: FeedForwardNetworkComponentQ
  } deriving (Show)

-- FSM states for autonomous stages
data WriteState = WriteIdle | WriteWriting | WriteDone
  deriving (Show, Eq, Generic, NFDataX)

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
      (kvBankController layerIndex processingState layerData qkvDone)
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
  xAfterAttn = residualAdder <$> layerData <*> woHeads
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

residualAdder :: LayerData -> Vec ModelDimension FixedPoint -> Vec ModelDimension FixedPoint
residualAdder layerData = zipWith (+) (inputVector layerData)

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

kvBankController ::
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
kvBankController layerIndex psSig idSig qkvValid (headOutAcc, headDoneAcc, writeDoneAcc) kvIx =
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
  (wrPulse, wrDone) = Cache.writePulseGenerator (isStage2Write .&&. qkvValid)
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


-- FSM states
data FSMState = IDLE | REQUESTING | PROJECTING | DONE
  deriving (Eq, Show, Generic, NFDataX)

{-|
  Ready/Valid Handshaking Protocol with internal backpressure control:

  • IDLE: Waiting for new input (validIn && readyOut)
  • REQUESTING: Sending request to multiplier until it accepts (woReadyOut)
  • PROJECTING: Multiplier is computing. We assert woReadyIn based on downstream readiness.
  • DONE: Output is valid, waiting to be consumed.
-}
singleHeadController :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool                              -- validIn (head done signal)
  -> Signal dom (Vec HeadDimension FixedPoint)    -- head vector
  -> MatI8E ModelDimension HeadDimension          -- WO matrix
  -> ( Signal dom (Vec ModelDimension FixedPoint) -- projected output
     , Signal dom Bool                            -- validOut
     , Signal dom Bool                            -- readyOut (can accept new head)
     )
singleHeadController validIn headVector woMatrix = (projOut, validOut, readyOut)
  where
    -- === FSM State ===
    state :: Signal dom FSMState
    state = register IDLE nextState

    -- === Handshakes ===
    upstreamHandshake         = validIn .&&. readyOut
    multiplierRequestHandshake = woValidIn .&&. woReadyOut
    multiplierResultHandshake  = woValidOut .&&. internalReady  -- now uses internal backpressure

    -- === State Transition Logic ===
    nextState = transition <$> state
                <*> upstreamHandshake
                <*> multiplierRequestHandshake
                <*> multiplierResultHandshake

    transition :: FSMState -> Bool -> Bool -> Bool -> FSMState
    transition IDLE upHS _ _
      | upHS      = REQUESTING
      | otherwise = IDLE

    transition REQUESTING _ reqHS _
      | reqHS     = PROJECTING
      | otherwise = REQUESTING

    transition PROJECTING _ _ resHS
      | resHS     = DONE
      | otherwise = PROJECTING

    transition DONE _ _ _ = IDLE

    -- === Ready signals ===
    readyOut :: Signal dom Bool
    readyOut = (==) <$> state <*> pure IDLE

    -- === Input latch ===
    latchedVector :: Signal dom (Vec HeadDimension FixedPoint)
    latchedVector = regEn (repeat 0) upstreamHandshake headVector

    -- === Multiplier interface ===
    woValidIn :: Signal dom Bool
    woValidIn = (==) <$> state <*> pure REQUESTING

    -- Internal backpressure: allow multiplier progress if computing OR consumer ready
    internalReady :: Signal dom Bool
    internalReady = mux (state .==. pure PROJECTING)
                        (pure True)   -- while computing, proceed freely
                        readyOut      -- when done, only accept if consumer ready

    -- Connect to matrix multiplier
    (woResult, woValidOut, woReadyOut) =
      matrixMultiplier woValidIn internalReady woMatrix latchedVector

    -- === Output latch and valid signal ===
    projOut :: Signal dom (Vec ModelDimension FixedPoint)
    projOut = regEn (repeat 0) multiplierResultHandshake woResult

    validOut :: Signal dom Bool
    validOut = (==) <$> state <*> pure DONE
