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
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Numeric.Operations (matrixMultiplier)

multiHeadAttentionStage :: forall dom.
  (HiddenClockResetEnable dom) =>
  MultiHeadAttentionComponentQ ->
  Signal dom ProcessingState ->
  Index NumLayers ->
  Signal dom LayerData ->
  ( Signal dom Bool,
    Signal dom (Vec ModelDimension FixedPoint),
    Signal
      dom
      ( Vec NumQueryHeads (Vec HeadDimension FixedPoint),
        Vec NumKeyValueHeads (Vec HeadDimension FixedPoint),
        Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
      ),
    Signal dom Bool,
    Signal dom Bool,
    Signal dom Bool
  )
multiHeadAttentionStage mha processingState layerIndex layerData = (attentionDone, xAfterAttn, qkvProjected, qkvInReady, writeDone, qkvDone)
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
    (qkvProjected, qkvValidOut, qkvInReady) =
      qkvProjectionController
        isStage1ThisLayer
        qkvOutReady
        input
        mha
        processingState
    qkvDone = qkvValidOut

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
        qkvValidOut -- validIn: start when QKV completes
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

-- FSM states
data FSMState = IDLE | REQUESTING | PROJECTING | DONE
  deriving (Eq, Show, Generic, NFDataX)

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
    state :: Signal dom FSMState
    state = register IDLE nextState

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

    transition :: FSMState -> Bool -> Bool -> Bool -> FSMState
    transition IDLE upHS _ _
      | upHS = REQUESTING
      | otherwise = IDLE
    transition REQUESTING _ reqHS _
      | reqHS = PROJECTING
      | otherwise = REQUESTING
    transition PROJECTING _ _ resHS
      | resHS = DONE
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
    internalReady =
      mux
        (state .==. pure PROJECTING)
        (pure True) -- while computing, proceed freely
        readyOut -- when done, only accept if consumer ready

    -- Connect to matrix multiplier
    (woResult, woValidOut, woReadyOut) =
      matrixMultiplier woValidIn internalReady woMatrix latchedVector

    -- === Output latch and valid signal ===
    projOut :: Signal dom (Vec ModelDimension FixedPoint)
    projOut = regEn (repeat 0) multiplierResultHandshake woResult

    validOut :: Signal dom Bool
    validOut = (==) <$> state <*> pure DONE

-- FSM states for autonomous stages
data WriteState = WriteIdle | WriteWriting | WriteDone
  deriving (Show, Eq, Generic, NFDataX)

kvWriteControllerFSM ::
  (HiddenClockResetEnable dom) =>
  Signal dom Bool -> -- validIn (QKV done)
  Signal dom Bool -> -- readyOut (Attn ready)
  Signal dom Bool -> -- writeComplete (all banks written)
  ( Signal dom Bool, -- validOut (write done, ready for attn)
    Signal dom Bool, -- readyIn (can accept QKV)
    Signal dom Bool -- enableWrite (trigger write)
  )
kvWriteControllerFSM validIn readyOut writeComplete = (validOut, readyIn, enableWrite)
  where
    state = register WriteIdle nextState

    readyIn = state .==. pure WriteIdle
    startWrite = validIn .&&. readyIn
    validOut = state .==. pure WriteDone
    consume = validOut .&&. readyOut

    nextState =
      mux
        (state .==. pure WriteIdle)
        (mux startWrite (pure WriteWriting) (pure WriteIdle))
        ( mux
            (state .==. pure WriteWriting)
            (mux writeComplete (pure WriteDone) (pure WriteWriting))
            (mux consume (pure WriteIdle) (pure WriteDone))
        )

    enableWrite = startWrite .||. (state .==. pure WriteWriting)
