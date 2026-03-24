module LLaMa2.Layer.Attention.MultiHeadAttention (
    multiHeadAttentionStage, singleHeadController
) where

import Clash.Prelude
import qualified GHC.TypeNats as TN
import LLaMa2.Types.LayerData (ActivationBramAddr)
import LLaMa2.Types.ModelConfig (ModelDimension, NumQueryHeads, HeadDimension, NumKeyValueHeads, SequenceLength, NumLayers, QHeadsPerKVBank)
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
  Signal dom FixedPoint                       ->  -- ^ bramRdData (BRAM slot 0/2, 1-cycle latency)
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
    (qkvAxiMaster, kvProjected, qBramWrites, _cosVec, _sinVec, qkvDone, qkvReady, qkvBramRdAddr) =
      qkvProjectionController
        cycleCounter qkvSlave layerIdx validIn writeReadyIn bramRdData seqPos

    (k, v) = unbundle kvProjected

    -- Distribute KV into per-head signals for KV banks.
    kPerHead = unbundle k  -- Vec NumKeyValueHeads (Signal dom (Vec HeadDimension FixedPoint))
    vPerHead = unbundle v  -- Vec NumKeyValueHeads (Signal dom (Vec HeadDimension FixedPoint))

    -------------------------------------------------------------------------
    -- Per-Q-head BRAMs (one per Q head, depth = HeadDimension)
    -- Written by QKV projection (qBramWrites), read by KV banks (qBramRdAddrs).
    -- blockRam's 1-cycle registered output breaks the circular dependency.
    -------------------------------------------------------------------------
    qhpk :: Int
    qhpk = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

    qBramRdDatas :: Vec NumQueryHeads (Signal dom FixedPoint)
    qBramRdDatas = imap (\qIdx _ ->
        blockRam (repeat 0 :: Vec HeadDimension FixedPoint)
                 (qBramRdAddrs !! qIdx)
                 (qBramWrites  !! qIdx)
      ) (repeat ())

    -- Flatten per-bank read addresses into a Vec NumQueryHeads.
    -- Bank kvIx owns Q heads [kvIx*qhpk .. kvIx*qhpk + qhpk - 1].
    qBramRdAddrs :: Vec NumQueryHeads (Signal dom (Index HeadDimension))
    qBramRdAddrs = imap (\qIdx _ ->
        let bank   = fromEnum qIdx `div` qhpk
            localJ = fromEnum qIdx `mod` qhpk
        in  (allBankQRdAddrs !! (toEnum bank   :: Index NumKeyValueHeads))
                              !! (toEnum localJ :: Index QHeadsPerKVBank)
      ) (repeat ())

    -- Per-bank Q BRAM read data slices (Vec QHeadsPerKVBank per bank).
    bankQBramData :: Index NumKeyValueHeads -> Vec QHeadsPerKVBank (Signal dom FixedPoint)
    bankQBramData kvIx = imap (\j _ ->
        qBramRdDatas !! (toEnum (fromEnum kvIx * qhpk + fromEnum j) :: Index NumQueryHeads)
      ) (repeat ())

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
          (bankQBramData kvIx)
          qkvDoneLatchedForWrite
          writeEnableForBanks
          attendActive
          kvIx
      ) (repeat () :: Vec NumKeyValueHeads ())

    kvAxiMasters = map (\(a,_,_,_,_) -> a) kvBankResultsVec
                :: Vec NumKeyValueHeads (Master.AxiMasterOut dom)

    allHeadOuts       = map (\(_,o,_,_,_) -> o) kvBankResultsVec
    allHeadDones      = map (\(_,_,d,_,_) -> d) kvBankResultsVec
    allWriteDone      = map (\(_,_,_,w,_) -> w) kvBankResultsVec
    allBankQRdAddrs   = map (\(_,_,_,_,a) -> a) kvBankResultsVec
                     :: Vec NumKeyValueHeads (Vec QHeadsPerKVBank (Signal dom (Index HeadDimension)))

    perHeadOutputs =
      fold (zipWith (liftA2 (zipWith (+)))) allHeadOuts
        :: Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))

    perHeadDoneFlags =
      fold (zipWith (liftA2 (||))) allHeadDones
        :: Vec NumQueryHeads (Signal dom Bool)

    perBankWriteDoneFlags = allWriteDone
      :: Vec NumKeyValueHeads (Signal dom Bool)

    -------------------------------------------------------------------------
    -- Latch each head's output at the moment its done flag rises.
    -- This ensures the vector is stable throughout the serial WO phase,
    -- even if the KV bank output changes after the head finishes.
    -------------------------------------------------------------------------
    perHeadDoneRise :: Vec NumQueryHeads (Signal dom Bool)
    perHeadDoneRise = map (\done -> done .&&. (not <$> register False done)) perHeadDoneFlags

    latchedHeadOutputs :: Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
    latchedHeadOutputs =
      zipWith (\doneRise out -> regEn (repeat 0) doneRise out) perHeadDoneRise perHeadOutputs

    -------------------------------------------------------------------------
    -- Serial WO head processing
    --
    -- Heads run one at a time in index order.
    -- Head 0 reads residual from slot 0; heads 1..N-1 accumulate into slot 2.
    -- Each head writes slot2[i] = rdBase[i] + projectedRow[i] to the BRAM.
    -- writeDone fires one cycle after the last head's outputValid pulse.
    -------------------------------------------------------------------------
    allHeadsDoneRise = allHeadsDone .&&. (not <$> register False allHeadsDone)

    -- woPhaseActive: True while any WO head is still processing.
    woPhaseActive = register False $
      mux allHeadsDoneRise (pure True) $
      mux lastWOHeadDone   (pure False)
      woPhaseActive

    -- headActive: index of the currently active WO head.
    headActive = register (0 :: Index NumQueryHeads) $
      mux anyWOOutputValid (satSucc SatWrap <$> headActive) headActive

    -- rdBase: slot 0 for head 0 (residual source), slot 2 for subsequent heads.
    rdBase = mux (headActive .==. pure 0) (pure slot0BramBase) (pure slot2BramBase)

    woResults = imap (\headIdx _ ->
        WOHP.woHeadProjector cycleCounter (perWOSlaves !! headIdx) layerIdx headIdx
          (woPhaseActive .&&. headActive .==. pure headIdx)  -- inputValid
          (pure True)                                         -- downStreamReady
          (perWOConsumeSignals !! headIdx)                    -- consumeSignal
          bramRdData
          rdBase
          (latchedHeadOutputs !! headIdx)
      ) (repeat () :: Vec NumQueryHeads ())
        :: Vec NumQueryHeads ( Master.AxiMasterOut dom
                             , Signal dom ActivationBramAddr
                             , Signal dom (Maybe (ActivationBramAddr, FixedPoint))
                             , Signal dom Bool   -- outputValid
                             , Signal dom Bool   -- readyForInput
                             )

    woAxiMasterPer = map (\(axi, _, _, _, _) -> axi) woResults
    perWOOutputValids :: Vec NumQueryHeads (Signal dom Bool)
    perWOOutputValids = map (\(_, _, _, ov, _) -> ov) woResults

    -- consumeSignal for each head = its own outputValid pulse.
    perWOConsumeSignals = perWOOutputValids

    anyWOOutputValid = or <$> sequenceA perWOOutputValids

    lastWOHeadDone = anyWOOutputValid .&&. headActive .==. pure maxBound

    writeDone = register False lastWOHeadDone

    -------------------------------------------------------------------------
    -- WO sub-arbiter (weights DRAM)
    -------------------------------------------------------------------------
    (woAxiMaster, perWOSlaves) = ARB.axiArbiterWithRouting woTopSlave woAxiMasterPer

    -------------------------------------------------------------------------
    -- BRAM interface: mux between QKV phase and WO phase.
    -- Both phases are strictly sequential so only one drives non-zero/non-Nothing.
    -------------------------------------------------------------------------
    -- Combine bramRdAddr from all WO heads (at most one non-zero at a time).
    woBramRdAddr = foldl1 (liftA2 (+)) (map (\(_, br, _, _, _) -> br) woResults)

    -- Combine bramWrite from all WO heads (at most one non-Nothing at a time).
    woBramWrite = foldl1 (liftA2 (<|>)) (map (\(_, _, bw, _, _) -> bw) woResults)

    -- Time-multiplex read address: QKV rmsNorm uses qkvBramRdAddr; WO phase uses woBramRdAddr.
    bramRdAddr = mux woPhaseActive woBramRdAddr qkvBramRdAddr

    bramWrite = mux woPhaseActive woBramWrite (pure Nothing)

    readyOut = qkvReady .&&. (not <$> woPhaseActive)


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
