-- File: LLaMa2/Layer/Attention/MultiHeadAttention.hs (add AXI parameters)
module LLaMa2.Layer.Attention.MultiHeadAttention (
    multiHeadAttentionStage, singleHeadController
) where

import Clash.Prelude
import qualified Simulation.Parameters as PARAM (MultiHeadAttentionComponentQ (..), DecoderParameters, TransformerLayerComponent (..))
import LLaMa2.Types.LayerData (LayerData (..))
import LLaMa2.Types.ModelConfig (ModelDimension, NumQueryHeads, HeadDimension, NumKeyValueHeads, SequenceLength, NumLayers)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layer.Attention.QKVProjection (qkvProjectionController)
import LLaMa2.Layer.Attention.KVCache (kvBankController)
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplier)
import LLaMa2.Layer.Attention.FSM (SingleHeadState (..), kvWriteControllerFSM)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import Simulation.Parameters (DecoderParameters(..))
import LLaMa2.Layer.Attention.QueryHeadProjector (QHeadDebugInfo)

multiHeadAttentionStage :: forall dom.
  (HiddenClockResetEnable dom) =>
  Signal dom (Unsigned 32) ->                     -- cycle counter  
  Slave.AxiSlaveIn dom ->                     -- DRAM interface
  Index NumLayers ->                          -- layer index
  PARAM.DecoderParameters ->
  Signal dom (Index SequenceLength) ->
  Signal dom LayerData ->
  Signal dom Bool ->  -- validIn
  (
    Master.AxiMasterOut dom,     -- AXI master out
    Signal dom (Vec ModelDimension FixedPoint),
    Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom Bool,
    Signal dom Bool,
    Signal dom Bool,
    Signal dom Bool     
    , QHeadDebugInfo dom
  )
multiHeadAttentionStage cycleCounter dramSlaveIn layerIdx params seqPos layerData validIn =
  (axiMasterOut, xAfterAttn, q, k, v, qkvReady, qkvDone, writeDone, attentionDone, debugInfo)
  where
    layerParams = modelLayers params !! layerIdx
    mhaParams = PARAM.multiHeadAttention layerParams

    -- Write-back controller
    allBanksDone = and <$> sequenceA perBankWriteDoneFlags
    (writeDone, writeReadyIn, writeEnable) =
      kvWriteControllerFSM
        qkvDone
        (pure True)  -- always ready (?)
        allBanksDone

    -- Latch qkvDone across the whole write
    qkvDoneHandshake = qkvDone .&&. writeReadyIn
    qkvDoneLatchedForWrite =
      register False
        ( mux qkvDoneHandshake (pure True)
        ( mux allBanksDone     (pure False) qkvDoneLatchedForWrite ) )

    writeEnableForBanks = writeEnable .&&. qkvDoneLatchedForWrite

    -- Attend-stage local enable (independent of controller’s internal flag)
    writeDonePrev      = register False writeDone
    writeCompletePulse = writeDone .&&. (not <$> writeDonePrev)

    allHeadsDone = and <$> sequenceA perHeadDoneFlags
    attendActive =
      register False
        ( mux writeCompletePulse (pure True)
        ( mux allHeadsDone       (pure False) attendActive ) )

    input = inputVector <$> layerData

    -- QKV projection with AXI
    (axiMasterOut, qkvProjected, qkvDone, qkvReady, debugInfo) =
      qkvProjectionController
        cycleCounter
        dramSlaveIn
        layerIdx
        validIn
        writeReadyIn
        input
        params
        seqPos

    (q, k, v) = unbundle qkvProjected

    -- Stage2/3
    initHeadOutputs   = repeat (pure (repeat 0))
    initHeadDone      = repeat (pure False)
    initWriteDone     = repeat (pure False)

    (perHeadOutputs, perHeadDoneFlags, perBankWriteDoneFlags) =
      foldl
        ( kvBankController seqPos layerData
                           qkvDoneLatchedForWrite
                           writeEnableForBanks
                           attendActive )
        (initHeadOutputs, initHeadDone, initWriteDone)
        indicesI

    -- WO projection
    (perHeadProjected, perHeadOutputValids, perHeadReadyForInputs) =
      perHeadWOController perHeadOutputs perHeadDoneFlags (PARAM.mWoQ mhaParams)

    validProjection proj valid _ready = mux valid proj (pure (repeat 0))
    gatedHeads =
      zipWith3 validProjection
        perHeadProjected
        perHeadOutputValids
        perHeadReadyForInputs

    -- Sum heads and produce “projection valid” when all outputs are valid
    woHeads        = foldl1 (zipWith (+)) <$> sequenceA gatedHeads
    validProjected = and <$> sequenceA perHeadOutputValids

    xAfterAttn = residualAdder <$> layerData <*> woHeads

    attentionDone =
      let prevValid = register False validProjected
       in validProjected .&&. (not <$> prevValid)

perHeadWOController ::
  forall dom.
  (HiddenClockResetEnable dom) =>
  Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint)) ->
  Vec NumQueryHeads (Signal dom Bool) ->
  Vec NumQueryHeads (MatI8E ModelDimension HeadDimension) ->
  ( Vec NumQueryHeads (Signal dom (Vec ModelDimension FixedPoint)),
    Vec NumQueryHeads (Signal dom Bool),  -- outputValid per head
    Vec NumQueryHeads (Signal dom Bool)   -- readyForInput per head
  )
perHeadWOController perHeadOutputs perHeadDoneFlags mWoQs =
  (perHeadProjected, perHeadOutputValids, perHeadReadyForInputs)
  where
    -- Drive each head only when it is both needed (done flag) and ready to accept input
    perHeadReadyForInputs = map (\(_, _, readyForInput) -> readyForInput) perHeadResults
    headInputValids       = zipWith (.&&.) perHeadDoneFlags perHeadReadyForInputs

    perHeadResults =
      zipWith3 singleHeadController headInputValids perHeadOutputs mWoQs

    perHeadProjected      = map (\(result, _, _) -> result)      perHeadResults
    perHeadOutputValids   = map (\(_, outputValid, _) -> outputValid) perHeadResults

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
singleHeadController inputValid headVector woMatrix = (projOut, outputValid, readyForInput)
  where
    state :: Signal dom SingleHeadState
    state = register SINGLE_HEAD_IDLE nextState
    upstreamHandshake = inputValid .&&. readyForInput
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
    readyForInput = (==) <$> state <*> pure SINGLE_HEAD_IDLE
    latchedVector = regEn (repeat 0) upstreamHandshake headVector
    woValidIn = (==) <$> state <*> pure SINGLE_HEAD_REQUESTING
    internalReady = mux (state .==. pure SINGLE_HEAD_PROJECTING) (pure True) readyForInput
    (woResult, woValidOut, woReadyOut) =
      parallelRowMatrixMultiplier woValidIn internalReady woMatrix latchedVector
    projOut = regEn (repeat 0) multiplierResultHandshake woResult
    outputValid = (==) <$> state <*> pure SINGLE_HEAD_DONE
