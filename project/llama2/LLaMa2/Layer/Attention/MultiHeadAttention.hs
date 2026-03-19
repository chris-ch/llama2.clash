module LLaMa2.Layer.Attention.MultiHeadAttention (
    multiHeadAttentionStage, singleHeadController
) where

import Clash.Prelude
import qualified GHC.TypeNats as TN
import LLaMa2.Types.LayerData (ActivationBramAddr)
import LLaMa2.Types.ModelConfig (ModelDimension, NumQueryHeads, HeadDimension, NumKeyValueHeads, SequenceLength, NumLayers)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layer.Attention.QKVProjection (qkvProjectionController)
import LLaMa2.Layer.Attention.KVCache (kvBankControllerDRAM)
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplier)
import LLaMa2.Layer.Attention.FSM (SingleHeadState (..), kvWriteControllerFSM)
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Arbiter as ARB
import qualified LLaMa2.Layer.Attention.WOHeadProjector as WOHP

-- Base addresses in activation BRAM.
slot0BramBase :: ActivationBramAddr
slot0BramBase = 0

slot2BramBase :: ActivationBramAddr
slot2BramBase = natToNum @(2 TN.* ModelDimension)

multiHeadAttentionStage :: forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  ) =>
  Signal dom (Unsigned 32)                    ->  -- ^ cycle counter
  Slave.AxiSlaveIn dom                        ->  -- ^ weights DRAM
  Vec NumKeyValueHeads (Slave.AxiSlaveIn dom) ->  -- ^ KV cache DRAM (one per KV head)
  Signal dom (Index NumLayers)                ->
  Signal dom (Index SequenceLength)           ->
  Signal dom Bool                             ->  -- ^ validIn
  Signal dom FixedPoint                       ->  -- ^ bramRdData (BRAM slot 0, 1-cycle latency)
  ( Master.AxiMasterOut dom                       -- ^ weights AXI master
  , Vec NumKeyValueHeads (Master.AxiMasterOut dom) -- ^ KV cache AXI masters
  , Signal dom Bool                               -- ^ writeDone (slot 2 fully written)
  , Signal dom Bool                               -- ^ readyOut (to upstream)
  , Signal dom ActivationBramAddr                 -- ^ bramRdAddr (drive to BRAM read port)
  , Signal dom (Maybe (ActivationBramAddr, FixedPoint)) -- ^ bramWrite (drive to BRAM write port)
  )
multiHeadAttentionStage cycleCounter dramSlaveIn kvDramSlaves layerIdx seqPos
                        validIn bramRdData =
  (axiMasterOut, kvAxiMasters, writeDone, readyOut, bramRdAddr, bramWrite)
  where
    -- -----------------------------------------------------------------------
    -- KV write-back controller
    -- -----------------------------------------------------------------------
    allBanksDone = and <$> sequenceA perBankWriteDoneFlags
    (kvcWriteDone, writeReadyIn, writeEnable) =
      kvWriteControllerFSM qkvDone (pure True) allBanksDone

    qkvDoneHandshake = qkvDone .&&. writeReadyIn
    qkvDoneLatchedForWrite =
      register False
        ( mux qkvDoneHandshake (pure True)
        ( mux allBanksDone     (pure False) qkvDoneLatchedForWrite ) )

    writeEnableForBanks = writeEnable .&&. qkvDoneLatchedForWrite

    kvcWriteDonePrev      = register False kvcWriteDone
    writeCompletePulse = kvcWriteDone .&&. (not <$> kvcWriteDonePrev)

    allHeadsDone = and <$> sequenceA perHeadDoneFlags
    attendActive =
      register False
        ( mux writeCompletePulse (pure True)
        ( mux allHeadsDone       (pure False) attendActive ) )

    -------------------------------------------------------------------------
    -- Top-level 2-master AXI arbiter for weights DRAM: QKV + WO
    -------------------------------------------------------------------------
    topAllMasters = qkvAxiMaster :> woAxiMaster :> Nil :: Vec 2 (Master.AxiMasterOut dom)
    (axiMasterOut, topAllSlaves) = ARB.axiArbiterWithRouting dramSlaveIn topAllMasters
    topAllSlaves' = topAllSlaves :: Vec 2 (Slave.AxiSlaveIn dom)

    qkvSlave   = topAllSlaves' !! (0 :: Index 2)
    woTopSlave = topAllSlaves' !! (1 :: Index 2)

    -------------------------------------------------------------------------
    -- QKV projection (weights DRAM, reads from BRAM slot 0)
    -------------------------------------------------------------------------
    (qkvAxiMaster, qkvProjected, qkvDone, qkvReady, qkvBramRdAddr) =
      qkvProjectionController
        cycleCounter qkvSlave layerIdx validIn writeReadyIn bramRdData seqPos

    (q, k, v) = unbundle qkvProjected

    -- Distribute into per-head signals for KV banks and WO projectors.
    qPerHead = unbundle q  -- Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
    kPerHead = unbundle k  -- Vec NumKeyValueHeads (Signal dom (Vec HeadDimension FixedPoint))
    vPerHead = unbundle v  -- Vec NumKeyValueHeads (Signal dom (Vec HeadDimension FixedPoint))

    -------------------------------------------------------------------------
    -- KV cache banks (one per KV head, each with its own DRAM slave)
    -------------------------------------------------------------------------
    kvBankResultsVec = imap (\kvIx _ ->
        kvBankControllerDRAM
          cycleCounter
          (kvDramSlaves !! kvIx)
          layerIdx seqPos
          (kPerHead !! kvIx)
          (vPerHead !! kvIx)
          qPerHead
          qkvDoneLatchedForWrite
          writeEnableForBanks
          attendActive
          kvIx
      ) (repeat () :: Vec NumKeyValueHeads ())

    kvAxiMasters = map (\(a,_,_,_) -> a) kvBankResultsVec
                :: Vec NumKeyValueHeads (Master.AxiMasterOut dom)

    allHeadOuts  = map (\(_,o,_,_) -> o) kvBankResultsVec
    allHeadDones = map (\(_,_,d,_) -> d) kvBankResultsVec
    allWriteDone = map (\(_,_,_,w) -> w) kvBankResultsVec

    perHeadOutputs =
      fold (zipWith (liftA2 (zipWith (+)))) allHeadOuts
        :: Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))

    perHeadDoneFlags =
      fold (zipWith (liftA2 (||))) allHeadDones
        :: Vec NumQueryHeads (Signal dom Bool)

    perBankWriteDoneFlags = allWriteDone
      :: Vec NumKeyValueHeads (Signal dom Bool)

    -------------------------------------------------------------------------
    -- WO projection (weights DRAM, one projector per Q-head)
    -------------------------------------------------------------------------
    woConsumeSignal = woAllValid

    woResults = imap (\headIdx _ ->
        WOHP.woHeadProjector cycleCounter (perWOSlaves !! headIdx) layerIdx headIdx
          (perHeadDoneFlags  !! headIdx)
          (pure True)
          woConsumeSignal
          (perHeadOutputs !! headIdx)
      ) (repeat () :: Vec NumQueryHeads ())
        :: Vec NumQueryHeads ( Master.AxiMasterOut dom
                             , Signal dom (Vec ModelDimension FixedPoint)
                             , Signal dom Bool
                             , Signal dom Bool
                             )

    woAxiMasterPer = map (\(axi, _, _, _) -> axi) woResults
    woVecs         = map (\(_, v', _, _)  -> v')  woResults
    woValids       = map (\(_, _, va, _)  -> va)  woResults

    -------------------------------------------------------------------------
    -- WO sub-arbiter (weights DRAM)
    -------------------------------------------------------------------------
    (woAxiMaster, perWOSlaves) = ARB.axiArbiterWithRouting woTopSlave woAxiMasterPer

    woAllValid = and <$> sequenceA woValids

    gatedHeads = zipWith (\va vec -> mux va vec (pure (repeat 0))) woValids woVecs
    -- woHeads :: Signal dom (Vec ModelDimension FixedPoint)
    woHeads    = foldl1 (zipWith (+)) <$> sequenceA gatedHeads

    attentionDone =
      let prevValid = register False woAllValid
       in woAllValid .&&. (not <$> prevValid)

    -- Snapshot woHeads at attentionDone so the residual FSM has a stable
    -- value throughout its 64-cycle duration (woHeads goes invalid once
    -- the WO projectors see their woConsumeSignal and reset their outputs).
    woHeadsCapture = regEn (repeat 0) attentionDone woHeads

    --------------------------------------------------------------------------
    -- Sequential residual add FSM
    --   Fires on rising edge of woAllValid (attentionDone).
    --   Reads slot 0 element-by-element (1-cycle BRAM latency).
    --   Writes slot2[i] = slot0[i] + woHeads[i] to BRAM.
    --   After ModelDimension + 2 cycles, writeDone fires.
    --------------------------------------------------------------------------
    resActive = register False $
      mux attentionDone (pure True) $
      mux resLoadAtMax  (pure False)
      resActive

    resLoadCounter = register 0 $
      mux resActive (satSucc SatWrap <$> resLoadCounter) (pure 0 :: Signal dom (Index ModelDimension))

    resLoadAtMax = resActive .&&. resLoadCounter .==. pure maxBound

    resDrain = register False resLoadAtMax

    writeDone = register False resDrain

    -- Slot 0 read address during the residual phase.
    resRdAddr = (slot0BramBase +) . fromIntegral <$> resLoadCounter

    -- Time-multiplex BRAM read port: QKV rmsNorm phase uses qkvBramRdAddr;
    -- residual phase uses resRdAddr. Phases are strictly sequential.
    bramRdAddr = mux resActive resRdAddr qkvBramRdAddr

    readyOut = qkvReady .&&. (not <$> resActive) .&&. (not <$> resDrain)

    -- Write path: one cycle behind the read (BRAM latency).
    prevResCounter = register (0 :: Index ModelDimension) resLoadCounter
    prevResActive  = register False resActive

    inWritePhase = prevResActive .||. resDrain

    woElem      = (!!) <$> woHeadsCapture <*> prevResCounter
    slot2WrAddr = (slot2BramBase +) . fromIntegral <$> prevResCounter

    bramWrite = mux inWritePhase
      (Just <$> ((,) <$> slot2WrAddr <*> ((+) <$> bramRdData <*> woElem)))
      (pure Nothing)


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
