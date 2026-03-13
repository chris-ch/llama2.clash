-- | DRAM-backed KV cache bank controller for one KV head.
--
-- Write phase: packs K and V vectors and writes them to KV-cache DRAM.
-- Attend phase: fetches K then V for each position 0..seqPos and
-- presents them to attentionHead one row at a time.
module LLaMa2.Layer.Attention.KVCacheBankController
  ( kvCacheBankController
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec, fixedPointVecParser, fixedPointVecPackerVec, axiNWordFetcher)
import LLaMa2.Memory.AxiWriteMaster (axiWriteMaster)
import LLaMa2.Memory.KVCacheLayout (kvCacheKAddress, kvCacheVAddress)
import LLaMa2.Layer.Attention.AttentionHead (attentionHead)

-- ---------------------------------------------------------------------------
-- State machine
-- ---------------------------------------------------------------------------

data KVBankState
  = KVBIdle
  | KVBWriteK       -- K write in progress
  | KVBWriteV       -- V write in progress (K done)
  | KVBAttendK      -- Issue K fetch request (1-cycle pulse into fetcher)
  | KVBAttendWaitK  -- Waiting for K DRAM data
  | KVBAttendV      -- Issue V fetch request
  | KVBAttendWaitV  -- Waiting for V DRAM data
  | KVBAttendStep   -- Assert stepEn to attentionHead for 1 cycle
  | KVBAttendDone   -- Last row processed
  deriving (Generic, NFDataX, Eq, Show)

-- ---------------------------------------------------------------------------
-- Bank controller
-- ---------------------------------------------------------------------------

kvCacheBankController :: forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  )
  => Signal dom (Unsigned 32)                        -- ^ cycle counter
  -> Slave.AxiSlaveIn dom                            -- ^ dedicated KV cache DRAM slave
  -> Signal dom (Index NumLayers)                    -- ^ layer index
  -> Index NumKeyValueHeads                          -- ^ KV head index (static)
  -> Signal dom (Index SequenceLength)               -- ^ current sequence position
  -> Signal dom Bool                                 -- ^ qkvValid
  -> Signal dom Bool                                 -- ^ enableWriteKV
  -> Signal dom Bool                                 -- ^ enableAttend
  -> Signal dom (Vec HeadDimension FixedPoint)       -- ^ K vector to write
  -> Signal dom (Vec HeadDimension FixedPoint)       -- ^ V vector to write
  -> Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))  -- ^ query vectors
  -> ( Master.AxiMasterOut dom
     , Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Signal dom Bool                               -- ^ write done
     )
kvCacheBankController _cycleCounter dramSlaveIn layerIdx kvHeadIdx seqPos
                       qkvValid enableWriteKV enableAttend keyVec valVec queries =
  (axiMaster, headOutputs, headDones, writeDone)
  where
    -- -----------------------------------------------------------------------
    -- State machine
    -- -----------------------------------------------------------------------
    state :: Signal dom KVBankState
    state = register KVBIdle nextState

    prevState :: Signal dom KVBankState
    prevState = register KVBIdle state

    risingEdge :: Signal dom KVBankState -> Signal dom Bool
    risingEdge s = (state .==. s) .&&. (not <$> (prevState .==. s))

    -- -----------------------------------------------------------------------
    -- Latched sequence position (captured at write start, stable through attend)
    -- -----------------------------------------------------------------------
    startWrite :: Signal dom Bool
    startWrite = state .==. pure KVBIdle .&&. enableWriteKV .&&. qkvValid

    latchedSeqPos :: Signal dom (Index SequenceLength)
    latchedSeqPos = regEn 0 startWrite seqPos

    -- -----------------------------------------------------------------------
    -- Row counter for attention phase
    -- -----------------------------------------------------------------------
    rowCounter :: Signal dom (Index SequenceLength)
    rowCounter = register 0 nextRowCounter

    isLastRow :: Signal dom Bool
    isLastRow = rowCounter .==. latchedSeqPos

    nextRowCounter =
      mux (state .==. pure KVBAttendStep .&&. (not <$> isLastRow))
          (rowCounter + 1)
      $ mux (state .==. pure KVBAttendDone .||. state .==. pure KVBIdle)
          (pure 0)
          rowCounter

    -- -----------------------------------------------------------------------
    -- Single write master (shared for K and V writes)
    -- -----------------------------------------------------------------------
    burstLen :: Unsigned 8
    burstLen = fromIntegral (natToNum @(WordsPerFPVec HeadDimension) - 1 :: Int)

    -- Latch K and V packed data at write start
    latchedKWords :: Signal dom (Vec (WordsPerFPVec HeadDimension) (BitVector 512))
    latchedKWords = regEn (repeat 0) startWrite (fixedPointVecPackerVec <$> keyVec)

    latchedVWords :: Signal dom (Vec (WordsPerFPVec HeadDimension) (BitVector 512))
    latchedVWords = regEn (repeat 0) startWrite (fixedPointVecPackerVec <$> valVec)

    -- Write beat counter (for multi-beat bursts)
    writeBeat :: Signal dom (Index (WordsPerFPVec HeadDimension))
    writeBeat = register 0 nextWriteBeat
    nextWriteBeat =
      mux (startKBurst .||. startVBurst) (pure 0)
      $ mux writeDataReady (writeBeat + 1)
        writeBeat

    isWritingK :: Signal dom Bool
    isWritingK = state .==. pure KVBWriteK

    -- One-cycle start pulses for write master
    startKBurst :: Signal dom Bool
    startKBurst = risingEdge (pure KVBWriteK)

    startVBurst :: Signal dom Bool
    startVBurst = risingEdge (pure KVBWriteV)

    startBurst :: Signal dom Bool
    startBurst = startKBurst .||. startVBurst

    writeAddr :: Signal dom (Unsigned 32)
    writeAddr = mux isWritingK
      ((\li sp -> kvCacheKAddress li kvHeadIdx sp) <$> layerIdx <*> latchedSeqPos)
      ((\li sp -> kvCacheVAddress li kvHeadIdx sp) <$> layerIdx <*> latchedSeqPos)

    writeDataBeat :: Signal dom (BitVector 512)
    writeDataBeat = mux isWritingK
      ((!!) <$> latchedKWords <*> writeBeat)
      ((!!) <$> latchedVWords <*> writeBeat)

    (writeMaster, writeDoneRaw, writeDataReady) =
      axiWriteMaster dramSlaveIn writeAddr (pure burstLen) startBurst writeDataBeat (pure True)

    -- writeDone fires when V write completes
    writeDone = writeDoneRaw .&&. (state .==. pure KVBWriteV)

    -- -----------------------------------------------------------------------
    -- Read masters: K fetcher and V fetcher
    -- -----------------------------------------------------------------------
    fetchKReq :: Signal dom Bool
    fetchKReq = risingEdge (pure KVBAttendK) .&&. kFetchReady

    fetchVReq :: Signal dom Bool
    fetchVReq = risingEdge (pure KVBAttendV) .&&. vFetchReady

    kFetchAddr :: Signal dom (Unsigned 32)
    kFetchAddr = (\li rc -> kvCacheKAddress li kvHeadIdx rc) <$> layerIdx <*> rowCounter

    vFetchAddr :: Signal dom (Unsigned 32)
    vFetchAddr = (\li rc -> kvCacheVAddress li kvHeadIdx rc) <$> layerIdx <*> rowCounter

    (kReadMaster, kWordsOut, kDataValid, kFetchReady, _kDbg) =
      axiNWordFetcher @dom @(WordsPerFPVec HeadDimension)
        dramSlaveIn fetchKReq kFetchAddr

    (vReadMaster, vWordsOut, vDataValid, vFetchReady, _vDbg) =
      axiNWordFetcher @dom @(WordsPerFPVec HeadDimension)
        dramSlaveIn fetchVReq vFetchAddr

    kRow :: Signal dom (Vec HeadDimension FixedPoint)
    kRow = fixedPointVecParser <$> kWordsOut

    vRow :: Signal dom (Vec HeadDimension FixedPoint)
    vRow = fixedPointVecParser <$> vWordsOut

    -- Latch K row when kDataValid fires (stable through V fetch)
    kRowLatched :: Signal dom (Vec HeadDimension FixedPoint)
    kRowLatched = regEn (repeat 0) kDataValid kRow

    -- -----------------------------------------------------------------------
    -- State transitions
    -- -----------------------------------------------------------------------
    nextState = step
      <$> state
      <*> (enableWriteKV .&&. qkvValid)
      <*> (writeDoneRaw .&&. isWritingK)     -- K write done
      <*> writeDone                           -- V write done
      <*> enableAttend
      <*> kFetchReady
      <*> kDataValid
      <*> vFetchReady
      <*> vDataValid
      <*> isLastRow

    step :: KVBankState -> Bool -> Bool -> Bool -> Bool
         -> Bool -> Bool -> Bool -> Bool -> Bool
         -> KVBankState
    step KVBIdle      wEn _    _    aEn _    _    _    _    _
      | wEn             = KVBWriteK
      | aEn             = KVBAttendK
      | otherwise       = KVBIdle
    step KVBWriteK    _   kDn  _    _   _    _    _    _    _
      | kDn             = KVBWriteV
      | otherwise       = KVBWriteK
    step KVBWriteV    _   _    vDn  _   _    _    _    _    _
      | vDn             = KVBIdle
      | otherwise       = KVBWriteV
    step KVBAttendK   _   _    _    _   kRdy _    _    _    _
      | kRdy            = KVBAttendWaitK
      | otherwise       = KVBAttendK
    step KVBAttendWaitK _ _    _    _   _    kDv  _    _    _
      | kDv             = KVBAttendV
      | otherwise       = KVBAttendWaitK
    step KVBAttendV   _   _    _    _   _    _    vRdy _    _
      | vRdy            = KVBAttendWaitV
      | otherwise       = KVBAttendV
    step KVBAttendWaitV _ _    _    _   _    _    _    vDv  _
      | vDv             = KVBAttendStep
      | otherwise       = KVBAttendWaitV
    step KVBAttendStep _  _    _    _   _    _    _    _    last'
      | last'           = KVBAttendDone
      | otherwise       = KVBAttendK
    step KVBAttendDone _  _    _    _   _    _    _    _    _
                        = KVBIdle

    -- -----------------------------------------------------------------------
    -- attentionHead inputs
    -- -----------------------------------------------------------------------
    stepEn :: Signal dom Bool
    stepEn = state .==. pure KVBAttendStep

    -- Clear softmax state only on the very first attend row
    -- (when entering KVBAttendK from KVBIdle, not from KVBAttendStep)
    clearSt :: Signal dom Bool
    clearSt = (state .==. pure KVBAttendK) .&&. (prevState .==. pure KVBIdle)

    -- -----------------------------------------------------------------------
    -- Per-query attention heads (GQA: multiple Q heads per KV bank)
    -- -----------------------------------------------------------------------
    qIdx0 = queryHeadIndex0 kvHeadIdx
    hasQ1 = hasSecondQueryHead kvHeadIdx
    qIdx1 = queryHeadIndex1 kvHeadIdx

    (out0, done0) = attentionHead clearSt stepEn (queries !! qIdx0) kRowLatched vRow isLastRow
    (out1, done1)
      | hasQ1    = attentionHead clearSt stepEn (queries !! qIdx1) kRowLatched vRow isLastRow
      | otherwise = (pure (repeat 0), pure False)

    initOuts  = repeat (pure (repeat 0)) :: Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
    initDones = repeat (pure False) :: Vec NumQueryHeads (Signal dom Bool)

    headOutputs0 = replace qIdx0 out0 initOuts
    headOutputs  = if hasQ1 then replace qIdx1 out1 headOutputs0 else headOutputs0

    headDones0   = replace qIdx0 done0 initDones
    headDones    = if hasQ1 then replace qIdx1 done1 headDones0 else headDones0

    -- -----------------------------------------------------------------------
    -- AXI master mux: write phase takes priority, then read phases
    -- -----------------------------------------------------------------------
    isWritePhase :: Signal dom Bool
    isWritePhase = (state .==. pure KVBWriteK) .||. (state .==. pure KVBWriteV)

    isKReadPhase :: Signal dom Bool
    isKReadPhase = (state .==. pure KVBAttendK) .||. (state .==. pure KVBAttendWaitK)

    axiMaster :: Master.AxiMasterOut dom
    axiMaster =
      Master.axiMasterMux isWritePhase
        writeMaster
        (Master.axiMasterMux isKReadPhase kReadMaster vReadMaster)

-- ---------------------------------------------------------------------------
-- GQA helpers
-- ---------------------------------------------------------------------------

queryHeadsPerKeyValueHead :: Int
queryHeadsPerKeyValueHead = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads

maxQueryHeadIndex :: Int
maxQueryHeadIndex = natToNum @NumQueryHeads - 1

baseQueryIndex :: Index NumKeyValueHeads -> Int
baseQueryIndex kvIx = fromEnum kvIx * queryHeadsPerKeyValueHead

hasSecondQueryHead :: Index NumKeyValueHeads -> Bool
hasSecondQueryHead kvIx =
  queryHeadsPerKeyValueHead >= 2 && (baseQueryIndex kvIx + 1 <= maxQueryHeadIndex)

queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 kvIx = toEnum (min maxQueryHeadIndex (baseQueryIndex kvIx))

queryHeadIndex1 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex1 kvIx =
  if hasSecondQueryHead kvIx
    then toEnum (baseQueryIndex kvIx + 1)
    else queryHeadIndex0 kvIx
