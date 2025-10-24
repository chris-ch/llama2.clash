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
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplier)
import LLaMa2.Layer.Attention.FSM (SingleHeadState (..), kvWriteControllerFSM)
import LLaMa2.Layer.Attention.WeightBuffer (QKVWeightBuffer(..))

multiHeadAttentionStage :: forall dom.
  (HiddenClockResetEnable dom) =>
  MultiHeadAttentionComponentQ ->
  Signal dom ProcessingState ->
  Index NumLayers ->
  Signal dom LayerData ->
  Signal dom QKVWeightBuffer ->
  Signal dom Bool ->
  ( Signal dom Bool,
    Signal dom (Vec ModelDimension FixedPoint),
    Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom Bool,
    Signal dom Bool,
    Signal dom Bool
  )
multiHeadAttentionStage mha processingState layerIndex layerData weightBuffer enable =
  (attentionDone, xAfterAttn, q, k, v, qkvInReady, writeDone, qkvDone)
  where
    isStage1ThisLayer =
      ( \ps ->
          processingStage ps == Stage1_ProjectQKV &&
          processingLayer ps == layerIndex
      ) <$> processingState
    qkvOutReady =
      ( \ps ->
          processingStage ps == Stage2_WriteKV &&
          processingLayer ps == layerIndex
      ) <$> processingState

    input = inputVector <$> layerData

    -- CHANGED: RAM-aware controller
    (qkvProjected, qkvDone, qkvInReady) =
      qkvProjectionController
        isStage1ThisLayer
        qkvOutReady
        input
        mha
        processingState
        weightBuffer
        enable

    (q, k, v) = unbundle qkvProjected

    -- Stage2/3 unchanged
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
        qkvDone
        (pure True)
        allBanksDone
    writeDone = writeValidOutNew

    -- WO projection unchanged
    (perHeadProjected, perHeadValidOuts, perHeadReadyOuts) =
      perHeadWOController perHeadOutputs perHeadDoneFlags (mWoQ mha)
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

singleHeadController ::
  forall dom.
  (HiddenClockResetEnable dom) =>
  Signal dom Bool ->
  Signal dom (Vec HeadDimension FixedPoint) ->
  MatI8E ModelDimension HeadDimension ->
  ( Signal dom (Vec ModelDimension FixedPoint),
    Signal dom Bool,
    Signal dom Bool
  )
singleHeadController validIn headVector woMatrix = (projOut, validOut, readyOut)
  where
    state :: Signal dom SingleHeadState
    state = register SINGLE_HEAD_IDLE nextState
    upstreamHandshake = validIn .&&. readyOut
    multiplierRequestHandshake = woValidIn .&&. woReadyOut
    multiplierResultHandshake = woValidOut .&&. internalReady
    nextState =
      transition
        <$> state
        <*> upstreamHandshake
        <*> multiplierRequestHandshake
        <*> multiplierResultHandshake
    transition SINGLE_HEAD_IDLE upHS _ _     | upHS = SINGLE_HEAD_REQUESTING
                                             | otherwise = SINGLE_HEAD_IDLE
    transition SINGLE_HEAD_REQUESTING _ req _| req = SINGLE_HEAD_PROJECTING
                                             | otherwise = SINGLE_HEAD_REQUESTING
    transition SINGLE_HEAD_PROJECTING _ _ res| res = SINGLE_HEAD_DONE
                                             | otherwise = SINGLE_HEAD_PROJECTING
    transition SINGLE_HEAD_DONE _ _ _        = SINGLE_HEAD_IDLE
    readyOut = (==) <$> state <*> pure SINGLE_HEAD_IDLE
    latchedVector = regEn (repeat 0) upstreamHandshake headVector
    woValidIn = (==) <$> state <*> pure SINGLE_HEAD_REQUESTING
    internalReady = mux (state .==. pure SINGLE_HEAD_PROJECTING) (pure True) readyOut
    (woResult, woValidOut, woReadyOut) =
      parallelRowMatrixMultiplier woValidIn internalReady woMatrix latchedVector
    projOut = regEn (repeat 0) multiplierResultHandshake woResult
    validOut = (==) <$> state <*> pure SINGLE_HEAD_DONE
