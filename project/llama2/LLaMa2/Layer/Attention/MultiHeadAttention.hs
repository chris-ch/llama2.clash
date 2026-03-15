module LLaMa2.Layer.Attention.MultiHeadAttention (
    multiHeadAttentionStage, singleHeadController
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData (..))
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

multiHeadAttentionStage :: forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  ) =>
  Signal dom (Unsigned 32)                ->  -- cycle counter
  Slave.AxiSlaveIn dom                    ->  -- weights DRAM
  Vec NumKeyValueHeads (Slave.AxiSlaveIn dom) ->  -- KV cache DRAM (one per KV head)
  Signal dom (Index NumLayers)            ->
  Signal dom (Index SequenceLength)       ->
  Signal dom LayerData                    ->
  Signal dom Bool                         ->  -- validIn
  ( Master.AxiMasterOut dom                   -- weights AXI master
  , Vec NumKeyValueHeads (Master.AxiMasterOut dom)  -- KV cache AXI masters
  , Signal dom (Vec ModelDimension FixedPoint)
  , Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
  , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  , Signal dom Bool
  , Signal dom Bool
  , Signal dom Bool
  , Signal dom Bool
  )
multiHeadAttentionStage cycleCounter dramSlaveIn kvDramSlaves layerIdx seqPos layerData validIn =
  ( axiMasterOut, kvAxiMasters
  , xAfterAttn, q, k, v
  , qkvReady, qkvDone, writeDone, attentionDone)
  where
    -- -----------------------------------------------------------------------
    -- Write-back controller (same as before)
    -- -----------------------------------------------------------------------
    allBanksDone = and <$> sequenceA perBankWriteDoneFlags
    (writeDone, writeReadyIn, writeEnable) =
      kvWriteControllerFSM qkvDone (pure True) allBanksDone

    qkvDoneHandshake = qkvDone .&&. writeReadyIn
    qkvDoneLatchedForWrite =
      register False
        ( mux qkvDoneHandshake (pure True)
        ( mux allBanksDone     (pure False) qkvDoneLatchedForWrite ) )

    writeEnableForBanks = writeEnable .&&. qkvDoneLatchedForWrite

    writeDonePrev      = register False writeDone
    writeCompletePulse = writeDone .&&. (not <$> writeDonePrev)

    allHeadsDone = and <$> sequenceA perHeadDoneFlags
    attendActive =
      register False
        ( mux writeCompletePulse (pure True)
        ( mux allHeadsDone       (pure False) attendActive ) )

    input = inputVector <$> layerData

    -------------------------------------------------------------------------
    -- Top-level 2-master AXI arbiter for weights DRAM: QKV + WO
    -------------------------------------------------------------------------
    topAllMasters :: Vec 2 (Master.AxiMasterOut dom)
    topAllMasters = qkvAxiMaster :> woAxiMaster :> Nil

    topAllSlaves :: Vec 2 (Slave.AxiSlaveIn dom)
    (axiMasterOut, topAllSlaves) = ARB.axiArbiterWithRouting dramSlaveIn topAllMasters

    qkvSlave   = topAllSlaves !! (0 :: Index 2)
    woTopSlave = topAllSlaves !! (1 :: Index 2)

    -------------------------------------------------------------------------
    -- QKV projection (weights DRAM)
    -------------------------------------------------------------------------
    (qkvAxiMaster, qkvProjected, qkvDone, qkvReady) =
      qkvProjectionController
        cycleCounter qkvSlave layerIdx validIn writeReadyIn input seqPos

    (q, k, v) = unbundle qkvProjected

    -------------------------------------------------------------------------
    -- KV cache banks (one per KV head, each with its own DRAM slave)
    -------------------------------------------------------------------------
    kvBankResultsVec :: Vec NumKeyValueHeads
      ( Master.AxiMasterOut dom
      , Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
      , Vec NumQueryHeads (Signal dom Bool)
      , Signal dom Bool
      )
    kvBankResultsVec = imap (\kvIx _ ->
        kvBankControllerDRAM
          cycleCounter
          (kvDramSlaves !! kvIx)
          layerIdx seqPos layerData
          qkvDoneLatchedForWrite
          writeEnableForBanks
          attendActive
          kvIx
      ) (repeat () :: Vec NumKeyValueHeads ())

    kvAxiMasters :: Vec NumKeyValueHeads (Master.AxiMasterOut dom)
    kvAxiMasters = map (\(a,_,_,_) -> a) kvBankResultsVec

    -- Combine non-overlapping per-bank outputs (sum for vectors, OR for bools)
    allHeadOuts  = map (\(_,o,_,_) -> o) kvBankResultsVec
    allHeadDones = map (\(_,_,d,_) -> d) kvBankResultsVec
    allWriteDone = map (\(_,_,_,w) -> w) kvBankResultsVec

    perHeadOutputs :: Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
    perHeadOutputs =
      fold (zipWith (liftA2 (zipWith (+)))) allHeadOuts

    perHeadDoneFlags :: Vec NumQueryHeads (Signal dom Bool)
    perHeadDoneFlags =
      fold (zipWith (liftA2 (||))) allHeadDones

    perBankWriteDoneFlags :: Vec NumKeyValueHeads (Signal dom Bool)
    perBankWriteDoneFlags = allWriteDone

    -------------------------------------------------------------------------
    -- WO projection (weights DRAM, one projector per Q-head)
    -------------------------------------------------------------------------
    woConsumeSignal = woAllValid

    woResults :: Vec NumQueryHeads ( Master.AxiMasterOut dom
                                   , Signal dom (Vec ModelDimension FixedPoint)
                                   , Signal dom Bool
                                   , Signal dom Bool
                                   )
    woResults = imap (\headIdx _ ->
        WOHP.woHeadProjector cycleCounter (perWOSlaves !! headIdx) layerIdx headIdx
          (perHeadDoneFlags  !! headIdx)
          (pure True)
          woConsumeSignal
          (perHeadOutputs !! headIdx)
      ) (repeat () :: Vec NumQueryHeads ())

    woAxiMasterPer = map (\(axi, _, _, _) -> axi) woResults
    woVecs         = map (\(_, v', _, _)  -> v')  woResults
    woValids       = map (\(_, _, va, _)  -> va)  woResults

    -------------------------------------------------------------------------
    -- WO 8-master sub-arbiter (weights DRAM)
    -------------------------------------------------------------------------
    (woAxiMaster, perWOSlaves) = ARB.axiArbiterWithRouting woTopSlave woAxiMasterPer

    woAllValid     = and <$> sequenceA woValids
    validProjected = woAllValid

    gatedHeads = zipWith (\va vec -> mux va vec (pure (repeat 0))) woValids woVecs
    woHeads    = foldl1 (zipWith (+)) <$> sequenceA gatedHeads

    xAfterAttn = residualAdder <$> layerData <*> woHeads

    attentionDone =
      let prevValid = register False validProjected
       in validProjected .&&. (not <$> prevValid)


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
