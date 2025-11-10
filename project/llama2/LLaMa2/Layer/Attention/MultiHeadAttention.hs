module LLaMa2.Layer.Attention.MultiHeadAttention (
    multiHeadAttentionStage, singleHeadController
) where
import Clash.Prelude
import qualified Simulation.Parameters as PARAM (MultiHeadAttentionComponentQ (..))
import LLaMa2.Types.LayerData (LayerData (..))
import LLaMa2.Types.ModelConfig (ModelDimension, NumQueryHeads, HeadDimension, NumKeyValueHeads, SequenceLength)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layer.Attention.QKVProjection (qkvProjectionController)
import LLaMa2.Layer.Attention.KVCache (kvBankController)
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplier)
import LLaMa2.Layer.Attention.FSM (SingleHeadState (..), kvWriteControllerFSM)
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer (QKVProjectionWeightBuffer(..))

multiHeadAttentionStage :: forall dom.
  (HiddenClockResetEnable dom) =>
  PARAM.MultiHeadAttentionComponentQ ->
  Signal dom (Index SequenceLength) ->
  Signal dom LayerData ->
  Signal dom QKVProjectionWeightBuffer ->
  Signal dom Bool ->  -- useRAM
  Signal dom Bool ->  -- validIn
  (
    Signal dom (Vec ModelDimension FixedPoint),
    Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),
    Signal dom Bool,
    Signal dom Bool,
    Signal dom Bool,
    Signal dom Bool
  )
multiHeadAttentionStage mha seqPos layerData weightBuffer useRAM validIn =
  (xAfterAttn, q, k, v, qkvReady, qkvDone, writeDone, attentionDone)
  where

    -- Pipeline-based ready/valid control
    -- The write FSM generates the proper ready signal for QKV output
    allBanksDone = and <$> sequenceA perBankWriteDoneFlags
    (writeDone, writeReadyIn, writeEnable) =
      kvWriteControllerFSM
        qkvDone
        (pure True)
        allBanksDone
    
    -- Latch qkvDone for the duration of the write operation
    -- The banks need both writeEnable AND qkvDone, but qkvDone de-asserts after handshake
    -- So we create writeEnableWithValid that combines them properly
    qkvDoneHandshake = qkvDone .&&. writeReadyIn
    qkvDoneLatchedForWrite = register False (mux qkvDoneHandshake (pure True) 
                                            (mux allBanksDone (pure False) qkvDoneLatchedForWrite))
    
    -- Provide stable write enable to banks (stays high during entire write operation)
    writeEnableForBanks = writeEnable .&&. qkvDoneLatchedForWrite

    -- ========================================================================
    -- ATTEND STAGE CONTROL
    -- ========================================================================
    -- Detect write completion (rising edge of writeDone)
    writeDonePrev = register False writeDone
    writeCompletePulse = writeDone .&&. (not <$> writeDonePrev)
    
    -- Latch: set when write completes, clear when attention completes
    -- This replaces the controller's enableAttend signal
    allHeadsDone = and <$> sequenceA perHeadDoneFlags
    attendActive = register False 
      (mux writeCompletePulse (pure True)
        (mux allHeadsDone (pure False) attendActive))

    input = inputVector <$> layerData

    (qkvProjected, qkvDone, qkvReady) =
      qkvProjectionController
        validIn
        writeReadyIn
        input
        mha
        seqPos
        weightBuffer
        useRAM

    (q, k, v) = unbundle qkvProjected

    initHeadOutputs = repeat (pure (repeat 0))
    initHeadDone = repeat (pure False)
    initWriteDone = repeat (pure False)
    
    -- Pass FSM's writeEnableForBanks to banks (includes latched qkvDone)
    -- The banks AND this with writeEnableForBanks, so both must be latched versions
    -- Use locally-generated attendActive instead of controller's enableAttend
    (perHeadOutputs, perHeadDoneFlags, perBankWriteDoneFlags) =
      foldl
        (kvBankController seqPos layerData qkvDoneLatchedForWrite 
                         writeEnableForBanks attendActive)
        (initHeadOutputs, initHeadDone, initWriteDone)
        indicesI
    
    -- WO projection
    (perHeadProjected, perHeadValidOuts, perHeadReadyOuts) =
      perHeadWOController perHeadOutputs perHeadDoneFlags (PARAM.mWoQ mha)
    gatedHeads =
      zipWith3
        (\proj valid ready -> mux ready proj (pure (repeat 0)))
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
