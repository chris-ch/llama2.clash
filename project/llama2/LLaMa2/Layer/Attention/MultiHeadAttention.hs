module LLaMa2.Layer.Attention.MultiHeadAttention (
    multiHeadAttentionStage, singleHeadController
) where

import Clash.Prelude
import LLaMa2.Types.Parameters (MultiHeadAttentionComponentQ (..))
import LLaMa2.Types.LayerData (LayerData (..), ProcessingState (..), CycleStage (..))
import LLaMa2.Types.ModelConfig (NumLayers, ModelDimension, NumQueryHeads, HeadDimension, NumKeyValueHeads)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layer.Attention.QKVProjection (qkvProjectionController)
import LLaMa2.Layer.Attention.KVCache (kvBankController)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplier)
import LLaMa2.Layer.Attention.FSM (SingleHeadState (..), kvWriteControllerFSM)

multiHeadAttentionStage :: forall dom.
  (HiddenClockResetEnable dom) =>
  MultiHeadAttentionComponentQ ->
  Signal dom ProcessingState ->
  Index NumLayers ->
  Signal dom LayerData ->
  Signal dom (RowI8E ModelDimension) ->  -- parsed weights
  Signal dom Bool ->                     -- stream valid
  ( Signal dom Bool,
    Signal dom (Vec ModelDimension FixedPoint),
    Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom Bool,
    Signal dom Bool,
    Signal dom Bool
  )
multiHeadAttentionStage mha processingState layerIndex layerData parsedWeights streamValid = 
  (attentionDone, xAfterAttn, q, k, v, qkvInReady, writeDone, qkvDone)
  where
    -- === Multi-Head Attention (MHA) Stages ===
    -- Stage1: QKV Projection
    isStage1ThisLayer =
      ( \ps ->
          processingStage ps
            == Stage1_ProjectQKV
            && processingLayer ps
            == layerIndex
      )
        <$> processingState
    qkvOutReady =
      ( \ps ->
          processingStage ps
            == Stage2_WriteKV
            && processingLayer ps
            == layerIndex
      )
        <$> processingState

    input = inputVector <$> layerData
    (qkvProjected, qkvDone, qkvInReady) =
      qkvProjectionController
        isStage1ThisLayer
        qkvOutReady
        input
        mha
        processingState
        parsedWeights
        streamValid
    (q, k, v) = unbundle qkvProjected
    -- Stage2/3: KV Cache and Attention
    initHeadOutputs = repeat (pure (repeat 0))
    initHeadDone = repeat (pure False)
    initWriteDone = repeat (pure False)
    (perHeadOutputs, perHeadDoneFlags, perBankWriteDoneFlags) =
      foldl
        (kvBankController layerIndex processingState layerData qkvDone)
        (initHeadOutputs, initHeadDone, initWriteDone)
        indicesI
    allBanksDone = and <$> sequenceA perBankWriteDoneFlags
    (writeValidOutNew, _writeReadyIn, _writeEnable) =
      kvWriteControllerFSM
        qkvDone -- validIn: start when QKV completes
        (pure True) -- readyOut: always ready (attention doesn't use FSM)
        allBanksDone -- writeComplete: all banks finished writing
    writeDone = writeValidOutNew

    -- Per-head WO projection
    (perHeadProjected, perHeadValidOuts, perHeadReadyOuts) =
      perHeadWOController perHeadOutputs perHeadDoneFlags (mWoQ mha)
    gatedHeads :: Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint))
    gatedHeads =
      zipWith3
        (\proj _valid ready -> mux ready proj (pure (repeat 0)))
        perHeadProjected
        perHeadValidOuts
        perHeadReadyOuts
    woHeads = foldl1 (zipWith (+)) <$> sequenceA gatedHeads
    validProjected = and <$> sequenceA perHeadReadyOuts
    xAfterAttn = residualAdder <$> layerData <*> woHeads
    attentionDone =
      let prevReady = register False validProjected
       in validProjected .&&. (not <$> prevReady)

perHeadWOController ::
  forall dom.
  (HiddenClockResetEnable dom) =>
  Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint)) ->
  Vec NumQueryHeads (Signal dom Bool) ->
  Vec NumQueryHeads (MatI8E ModelDimension HeadDimension) ->
  ( Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint)),
    Vec NumQueryHeads (Signal dom Bool),
    Vec NumQueryHeads (Signal dom Bool)
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

-- |
--  Ready/Valid Handshaking Protocol with internal backpressure control:
--
--  • IDLE: Waiting for new input (validIn && readyOut)
--  • REQUESTING: Sending request to multiplier until it accepts (woReadyOut)
--  • PROJECTING: Multiplier is computing. We assert woReadyIn based on downstream readiness.
--  • DONE: Output is valid, waiting to be consumed.
singleHeadController ::
  forall dom.
  (HiddenClockResetEnable dom) =>
  Signal dom Bool -> -- validIn (head done signal)
  Signal dom (Vec HeadDimension FixedPoint) -> -- head vector
  MatI8E ModelDimension HeadDimension -> -- WO matrix
  ( Signal dom (Vec ModelDimension FixedPoint), -- projected output
    Signal dom Bool, -- validOut
    Signal dom Bool -- readyOut (can accept new head)
  )
singleHeadController validIn headVector woMatrix = (projOut, validOut, readyOut)
  where
    -- === FSM State ===
    state :: Signal dom SingleHeadState
    state = register SINGLE_HEAD_IDLE nextState

    -- === Handshakes ===
    upstreamHandshake = validIn .&&. readyOut
    multiplierRequestHandshake = woValidIn .&&. woReadyOut
    multiplierResultHandshake = woValidOut .&&. internalReady -- now uses internal backpressure

    -- === State Transition Logic ===
    nextState =
      transition
        <$> state
        <*> upstreamHandshake
        <*> multiplierRequestHandshake
        <*> multiplierResultHandshake

    transition :: SingleHeadState -> Bool -> Bool -> Bool -> SingleHeadState
    transition SINGLE_HEAD_IDLE upHS _ _
      | upHS = SINGLE_HEAD_REQUESTING
      | otherwise = SINGLE_HEAD_IDLE
    transition SINGLE_HEAD_REQUESTING _ reqHS _
      | reqHS = SINGLE_HEAD_PROJECTING
      | otherwise = SINGLE_HEAD_REQUESTING
    transition SINGLE_HEAD_PROJECTING _ _ resHS
      | resHS = SINGLE_HEAD_DONE
      | otherwise = SINGLE_HEAD_PROJECTING
    transition SINGLE_HEAD_DONE _ _ _ = SINGLE_HEAD_IDLE

    -- === Ready signals ===
    readyOut :: Signal dom Bool
    readyOut = (==) <$> state <*> pure SINGLE_HEAD_IDLE

    -- === Input latch ===
    latchedVector :: Signal dom (Vec HeadDimension FixedPoint)
    latchedVector = regEn (repeat 0) upstreamHandshake headVector

    -- === Multiplier interface ===
    woValidIn :: Signal dom Bool
    woValidIn = (==) <$> state <*> pure SINGLE_HEAD_REQUESTING

    -- Internal backpressure: allow multiplier progress if computing OR consumer ready
    internalReady :: Signal dom Bool
    internalReady =
      mux
        (state .==. pure SINGLE_HEAD_PROJECTING)
        (pure True) -- while computing, proceed freely
        readyOut -- when done, only accept if consumer ready

    -- Connect to matrix multiplier
    (woResult, woValidOut, woReadyOut) =
      parallelRowMatrixMultiplier woValidIn internalReady woMatrix latchedVector

    -- === Output latch and valid signal ===
    projOut :: Signal dom (Vec ModelDimension FixedPoint)
    projOut = regEn (repeat 0) multiplierResultHandshake woResult

    validOut :: Signal dom Bool
    validOut = (==) <$> state <*> pure SINGLE_HEAD_DONE
