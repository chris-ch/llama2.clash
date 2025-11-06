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

-- | Self-contained multi-head attention stage
-- Internally manages: QKV projection → KV cache write → Attention computation
-- External interface: enableAttention in, attentionDone out
multiHeadAttentionStage :: forall dom.
  (HiddenClockResetEnable dom) =>
  PARAM.MultiHeadAttentionComponentQ ->
  Signal dom (Index SequenceLength) ->
  Signal dom LayerData ->
  Signal dom QKVProjectionWeightBuffer ->
  Signal dom Bool ->  -- useRAM (weights loaded)
  Signal dom Bool ->  -- enableAttention (external enable from SequenceController)
  ( Signal dom Bool,  -- attentionDone (entire attention mechanism complete)
    Signal dom (Vec ModelDimension FixedPoint),  -- attention output (immediate, for storage)
    Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint)),  -- Q vectors (latched for introspection)
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)),  -- K vectors (latched for introspection)
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))   -- V vectors (latched for introspection)
  )
multiHeadAttentionStage mha seqPos layerData weightBuffer useRAM enableAttention =
  (attentionDone, xAfterAttn, qLatched, kLatched, vLatched)  -- Return immediate xAfterAttn, not latched
  where
    input = inputVector <$> layerData

    -- ==========================================================================
    -- Internal Stage 1: QKV Projection
    -- ==========================================================================
    -- QKV starts when enableAttention is high and we're ready
    -- QKV outputs are ready when qkvDone goes high
    
    qkvOutReady :: Signal dom Bool
    qkvOutReady = pure True  -- KV write controller will handle backpressure

    (qkvProjected, qkvDone, qkvInReady) =
      qkvProjectionController
        enableAttention  -- Start QKV when attention is enabled
        qkvOutReady      -- Always ready to accept QKV output
        input
        mha
        seqPos
        weightBuffer
        useRAM

    (q, k, v) = unbundle qkvProjected

    -- CRITICAL: Latch Q/K/V when qkvDone pulses (for introspection)
    -- These values need to be captured immediately before they're lost
    qLatched = regEn (repeat (repeat 0)) qkvDone q
    kLatched = regEn (repeat (repeat 0)) qkvDone k
    vLatched = regEn (repeat (repeat 0)) qkvDone v

    -- ==========================================================================
    -- Internal Stage 2: KV Cache Write
    -- ==========================================================================
    -- Write starts automatically when QKV completes
    -- Write is complete when all banks have written
    
    initHeadOutputs = repeat (pure (repeat 0))
    initHeadDone = repeat (pure False)
    initWriteDone = repeat (pure False)
    
    (perHeadOutputs, perHeadDoneFlags, perBankWriteDoneFlags) =
      foldl
        (kvBankController seqPos layerData qkvDone 
                         writeEnable attendEnable)  -- Internal enables derived from FSM
        (initHeadOutputs, initHeadDone, initWriteDone)
        indicesI
    
    allBanksDone = and <$> sequenceA perBankWriteDoneFlags
    
    -- KV write FSM: manages transition from QKV → Write → Attend
    (writeValidOut, writeReadyIn, writeEnable) =
      kvWriteControllerFSM
        qkvDone          -- Start write when QKV completes
        (pure True)      -- Attention stage always ready to consume write completion
        allBanksDone     -- Write complete when all banks done

    writeDone = writeValidOut

    -- ==========================================================================
    -- Internal Stage 3: Attention Computation
    -- ==========================================================================
    -- Attention starts automatically when write completes
    -- Attention computation uses the per-head outputs from KV cache
    
    attendEnable = writeDone  -- Start attention when write completes

    -- WO projection (per-head processing)
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
    
    -- Add residual connection
    xAfterAttn = residualAdder <$> layerData <*> woHeads
    
    -- CRITICAL: Define edge detector first (outer scope)
    prevReady = register True validProjected  -- start true -> suppress spurious initial rising edge
    attentionDonePulse = validProjected .&&. (not <$> prevReady)
    
    -- Attention done: rising edge of validProjected
    -- This signals completion of the entire attention mechanism
    attentionDone = attentionDonePulse

-- | Per-head WO projection controller (unchanged)
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

-- | Residual connection (unchanged)
residualAdder :: LayerData -> Vec ModelDimension FixedPoint -> Vec ModelDimension FixedPoint
residualAdder layerData = zipWith (+) (inputVector layerData)

-- | Single head WO projection controller (unchanged)
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
