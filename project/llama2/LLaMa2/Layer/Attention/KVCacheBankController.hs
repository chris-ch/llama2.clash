-- | DRAM-backed KV cache bank controller for one KV head.
--
-- Write phase: packs K and V vectors and writes them to KV-cache DRAM.
-- Attend phase: fetches K, then serially reads Q from BRAM and computes
-- the rotary-encoded Q·K dot product, then fetches V and presents
-- to attentionHead one row at a time.
module LLaMa2.Layer.Attention.KVCacheBankController
  ( kvCacheBankController
  ) where

import Clash.Prelude
import Data.Maybe (isJust)
import qualified GHC.TypeNats as TN

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec, fixedPointVecParser, axiNWordFetcher)
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
  | KVBQDot0        -- Serial Q·K dot product for Q head 0
  | KVBQDot1        -- Serial Q·K dot product for Q head 1 (GQA only)
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
  => Signal dom (Unsigned 32)                                         -- ^ cycle counter
  -> Slave.AxiSlaveIn dom                                             -- ^ dedicated KV cache DRAM slave
  -> Signal dom (Index NumLayers)                                     -- ^ layer index
  -> Index NumKeyValueHeads                                           -- ^ KV head index (static)
  -> Signal dom (Index SequenceLength)                                -- ^ current sequence position
  -> Signal dom Bool                                                  -- ^ qkvValid
  -> Signal dom Bool                                                  -- ^ enableWriteKV
  -> Signal dom Bool                                                  -- ^ enableAttend
  -> Signal dom (Maybe (Index HeadDimension, FixedPoint))             -- ^ K element writes (with rotary, streaming)
  -> Signal dom (Maybe (Index HeadDimension, FixedPoint))             -- ^ V element writes (raw, streaming)
  -> Vec QHeadsPerKVBank (Signal dom FixedPoint)                      -- ^ Q BRAM read data (1-cycle latency)
  -> ( Master.AxiMasterOut dom
     , Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Signal dom Bool                               -- ^ write done
     , Vec QHeadsPerKVBank (Signal dom (Index HeadDimension))         -- ^ Q BRAM read addresses
     )
kvCacheBankController _cycleCounter dramSlaveIn layerIdx kvHeadIdx seqPos
                       qkvValid enableWriteKV enableAttend kElemWrite vElemWrite
                       qBramRdDatas =
  (axiMaster, headOutputs, headDones, writeDone, qBramRdAddrs)
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

    -- -----------------------------------------------------------------------
    -- Streaming K/V element → BRAM word packing
    --
    -- Elements arrive one at a time from KeyValueHeadProjector as
    -- Maybe (Index HeadDimension, FixedPoint).  We pack them into 512-bit
    -- AXI words using the same little-endian layout as fixedPointVecPackerVec:
    --   element i → word (i/16), bytes [(i%16)*4 .. (i%16)*4+3].
    -- A completed word is written to the K (or V) BRAM so that the write
    -- master can stream beats directly from BRAM.
    -- -----------------------------------------------------------------------
    kElemValid :: Signal dom Bool
    kElemValid = isJust <$> kElemWrite

    kElemIdx :: Signal dom (Index HeadDimension)
    kElemIdx = (\w -> case w of { Just (i,_) -> i; Nothing -> 0 }) <$> kElemWrite

    kElemVal :: Signal dom FixedPoint
    kElemVal = (\w -> case w of { Just (_,v) -> v; Nothing -> 0 }) <$> kElemWrite

    vElemValid :: Signal dom Bool
    vElemValid = isJust <$> vElemWrite

    vElemIdx :: Signal dom (Index HeadDimension)
    vElemIdx = (\w -> case w of { Just (i,_) -> i; Nothing -> 0 }) <$> vElemWrite

    vElemVal :: Signal dom FixedPoint
    vElemVal = (\w -> case w of { Just (_,v) -> v; Nothing -> 0 }) <$> vElemWrite

    -- Word-complete when the element is the last in a 512-bit word (position 15)
    -- OR is the last element overall (maxBound of HeadDimension).
    wordComplete :: Signal dom (Index HeadDimension) -> Signal dom Bool -> Signal dom Bool
    wordComplete idx valid = valid .&&.
      ((\i -> fromEnum i `mod` 16 == 15 || i == maxBound) <$> idx)

    wordBramIdx :: Signal dom (Index HeadDimension)
               -> Signal dom (Index (WordsPerFPVec HeadDimension))
    wordBramIdx idx = toEnum . (`div` 16) . fromEnum <$> idx

    -- Insert one FixedPoint into a 512-bit word buffer at the correct byte
    -- position, matching the little-endian layout of fixedPointVecPackerVec.
    insertFP :: Index HeadDimension -> FixedPoint -> BitVector 512 -> BitVector 512
    insertFP idx fp word =
      let bits :: BitVector 32
          bits      = pack fp
          bytePos   = (fromEnum idx `mod` 16) * 4
          bytes     = unpack word :: Vec 64 (BitVector 8)
          b0        = resize bits              :: BitVector 8
          b1        = resize (bits `shiftR`  8) :: BitVector 8
          b2        = resize (bits `shiftR` 16) :: BitVector 8
          b3        = resize (bits `shiftR` 24) :: BitVector 8
          bytes'    = replace (fromIntegral bytePos       :: Index 64) b0
                    $ replace (fromIntegral (bytePos + 1) :: Index 64) b1
                    $ replace (fromIntegral (bytePos + 2) :: Index 64) b2
                    $ replace (fromIntegral (bytePos + 3) :: Index 64) b3
                    bytes
      in pack bytes'

    -- K word buffer: accumulates elements until a full 512-bit word is ready.
    kWordBufNext :: Signal dom (BitVector 512)
    kWordBufNext = mux kElemValid (insertFP <$> kElemIdx <*> kElemVal <*> kWordBuf) kWordBuf

    kWordBuf :: Signal dom (BitVector 512)
    kWordBuf = register 0 $ mux (wordComplete kElemIdx kElemValid) (pure 0) kWordBufNext

    kWordBramWr :: Signal dom (Maybe (Index (WordsPerFPVec HeadDimension), BitVector 512))
    kWordBramWr = mux (wordComplete kElemIdx kElemValid)
      (curry Just <$> wordBramIdx kElemIdx <*> kWordBufNext)
      (pure Nothing)

    -- V word buffer
    vWordBufNext :: Signal dom (BitVector 512)
    vWordBufNext = mux vElemValid (insertFP <$> vElemIdx <*> vElemVal <*> vWordBuf) vWordBuf

    vWordBuf :: Signal dom (BitVector 512)
    vWordBuf = register 0 $ mux (wordComplete vElemIdx vElemValid) (pure 0) vWordBufNext

    vWordBramWr :: Signal dom (Maybe (Index (WordsPerFPVec HeadDimension), BitVector 512))
    vWordBramWr = mux (wordComplete vElemIdx vElemValid)
      (curry Just <$> wordBramIdx vElemIdx <*> vWordBufNext)
      (pure Nothing)

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
      ((`kvCacheKAddress` kvHeadIdx) <$> layerIdx <*> latchedSeqPos)
      ((`kvCacheVAddress` kvHeadIdx) <$> layerIdx <*> latchedSeqPos)

    -- K and V AXI-word BRAMs (depth = WordsPerFPVec HeadDimension).
    -- Read address = nextWriteBeat (1 cycle ahead) so data is ready when needed.
    kWordBramOut :: Signal dom (BitVector 512)
    kWordBramOut = blockRam (repeat 0 :: Vec (WordsPerFPVec HeadDimension) (BitVector 512))
                             nextWriteBeat kWordBramWr

    vWordBramOut :: Signal dom (BitVector 512)
    vWordBramOut = blockRam (repeat 0 :: Vec (WordsPerFPVec HeadDimension) (BitVector 512))
                             nextWriteBeat vWordBramWr

    writeDataBeat :: Signal dom (BitVector 512)
    writeDataBeat = mux isWritingK kWordBramOut vWordBramOut

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
    kFetchAddr = (`kvCacheKAddress` kvHeadIdx) <$> layerIdx <*> rowCounter

    vFetchAddr :: Signal dom (Unsigned 32)
    vFetchAddr = (`kvCacheVAddress` kvHeadIdx) <$> layerIdx <*> rowCounter


    (kReadMaster, kWordsOut, kDataValid, kFetchReady, _kDbg, _, _, _) =
      axiNWordFetcher @dom @(WordsPerFPVec HeadDimension)
        dramSlaveIn fetchKReq kFetchAddr

    (vReadMaster, vWordsOut, vDataValid, vFetchReady, _vDbg, _, _, _) =
      axiNWordFetcher @dom @(WordsPerFPVec HeadDimension)
        dramSlaveIn fetchVReq vFetchAddr

    kRow :: Signal dom (Vec HeadDimension FixedPoint)
    kRow = fixedPointVecParser <$> kWordsOut

    vRow :: Signal dom (Vec HeadDimension FixedPoint)
    vRow = fixedPointVecParser <$> vWordsOut

    -- Latch K row when kDataValid fires (stable through Q-dot and V fetch)
    kRowLatched :: Signal dom (Vec HeadDimension FixedPoint)
    kRowLatched = regEn (repeat 0) kDataValid kRow

    -- -----------------------------------------------------------------------
    -- Serial Q·K dot product with inline rotary encoding
    -- -----------------------------------------------------------------------

    -- Counter: 0..HeadDimension (HeadDimension+1 states)
    -- Cycle C: issue BRAM read addr C (if C < HeadDimension); receive data[C-1] (if C >= 1)
    qDotCounter :: Signal dom (Index (HeadDimension TN.+ 1))
    qDotCounter = register 0 nextQDotCounter

    inQDot0 :: Signal dom Bool
    inQDot0 = state .==. pure KVBQDot0

    inQDot1 :: Signal dom Bool
    inQDot1 = state .==. pure KVBQDot1

    inQDot :: Signal dom Bool
    inQDot = inQDot0 .||. inQDot1

    enterQDot0 :: Signal dom Bool
    enterQDot0 = risingEdge (pure KVBQDot0)

    enterQDot1 :: Signal dom Bool
    enterQDot1 = risingEdge (pure KVBQDot1)

    nextQDotCounter =
      mux (enterQDot0 .||. enterQDot1) (pure 0)
      $ mux (inQDot .&&. qDotCounter ./=. pure maxBound) (qDotCounter + 1)
        qDotCounter

    -- qDotDone must not fire on the entry cycle: the counter reset to 0 only
    -- takes effect the *next* cycle, so without this guard qDotDone would fire
    -- spuriously on the first cycle of every QDot phase.
    qDotDone :: Signal dom Bool
    qDotDone = inQDot .&&. (qDotCounter .==. pure maxBound)
                      .&&. (not <$> (enterQDot0 .||. enterQDot1))

    -- Q BRAM read address (used by both QDot0 and QDot1)
    qDotRdAddr :: Signal dom (Index HeadDimension)
    qDotRdAddr = (\c ->
        let v = fromEnum c
        in  toEnum (if v < natToNum @HeadDimension then v else natToNum @HeadDimension - 1)
      ) <$> qDotCounter

    -- All Q BRAMs driven with same address (only active bank's data is used)
    qBramRdAddrs :: Vec QHeadsPerKVBank (Signal dom (Index HeadDimension))
    qBramRdAddrs = repeat qDotRdAddr

    -- Select active BRAM data (head 0 during QDot0, head 1 during QDot1)
    activeBramData :: Signal dom FixedPoint
    activeBramData = mux inQDot1 (last qBramRdDatas) (head qBramRdDatas)

    -- Even/odd detection on qDotCounter:
    -- At cycle C, BRAM output corresponds to addr C-1.
    -- C is odd  (1,3,...) → data index C-1 is even → latch as qEven
    -- C is even (2,4,...) and C>=2 → data index C-1 is odd → compute contribution
    isEvenDataCycle :: Signal dom Bool
    isEvenDataCycle = inQDot .&&. (odd . fromEnum <$> qDotCounter)

    isOddDataCycle :: Signal dom Bool
    isOddDataCycle = inQDot .&&. (even . fromEnum <$> qDotCounter)
                            .&&. (qDotCounter .>. pure 0)

    -- Latch q_even when even data arrives
    latchedQEven :: Signal dom FixedPoint
    latchedQEven = regEn 0 isEvenDataCycle activeBramData

    -- K element indices for this pair
    -- At cycle C (even, >=2): k_even = k[C-2], k_odd = k[C-1]
    kEvenIdx :: Signal dom (Index HeadDimension)
    kEvenIdx = (\c -> toEnum (max 0 (fromEnum c - 2))) <$> qDotCounter

    kOddIdx :: Signal dom (Index HeadDimension)
    kOddIdx  = (\c -> toEnum (max 0 (fromEnum c - 1))) <$> qDotCounter

    kEval :: Signal dom FixedPoint
    kEval = (!!) <$> kRowLatched <*> kEvenIdx

    kOval :: Signal dom FixedPoint
    kOval = (!!) <$> kRowLatched <*> kOddIdx

    -- Dot-product contribution for this pair.
    -- Both Q (from Q BRAM) and K (from KV cache) are already rotary-encoded
    -- with their absolute positions, so their inner product equals the correct
    -- relative-position RoPE score:
    --   (R(q_pos)*q_raw) · (R(k_pos)*k_raw) = q_raw · R(k_pos - q_pos) · k_raw
    -- No additional rotation is needed here.
    contribution :: Signal dom FixedPoint
    contribution = liftA2 (+) (liftA2 (*) latchedQEven kEval)
                               (liftA2 (*) activeBramData kOval)

    -- Accumulators for the two possible Q heads
    dotAcc0 :: Signal dom FixedPoint
    dotAcc0 = register 0 $
      mux enterQDot0 (pure 0) $
      mux (inQDot0 .&&. isOddDataCycle) (liftA2 (+) dotAcc0 contribution)
      dotAcc0

    dotAcc1 :: Signal dom FixedPoint
    dotAcc1 = register 0 $
      mux enterQDot1 (pure 0) $
      mux (inQDot1 .&&. isOddDataCycle) (liftA2 (+) dotAcc1 contribution)
      dotAcc1

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
      <*> qDotDone

    step :: KVBankState -> Bool -> Bool -> Bool -> Bool
         -> Bool -> Bool -> Bool -> Bool -> Bool -> Bool
         -> KVBankState
    step KVBIdle      wEn _    _    aEn _    _    _    _    _    _
      | wEn             = KVBWriteK
      | aEn             = KVBAttendK
      | otherwise       = KVBIdle
    step KVBWriteK    _   kDn  _    _   _    _    _    _    _    _
      | kDn             = KVBWriteV
      | otherwise       = KVBWriteK
    step KVBWriteV    _   _    vDn  _   _    _    _    _    _    _
      | vDn             = KVBIdle
      | otherwise       = KVBWriteV
    step KVBAttendK   _   _    _    _   kRdy _    _    _    _    _
      | kRdy            = KVBAttendWaitK
      | otherwise       = KVBAttendK
    step KVBAttendWaitK _ _    _    _   _    kDv  _    _    _    _
      | kDv             = KVBQDot0
      | otherwise       = KVBAttendWaitK
    step KVBQDot0     _   _    _    _   _    _    _    _    _    qDone
      | qDone           = if hasQ1 then KVBQDot1 else KVBAttendV
      | otherwise       = KVBQDot0
    step KVBQDot1     _   _    _    _   _    _    _    _    _    qDone
      | qDone           = KVBAttendV
      | otherwise       = KVBQDot1
    step KVBAttendV   _   _    _    _   _    _    vRdy _    _    _
      | vRdy            = KVBAttendWaitV
      | otherwise       = KVBAttendV
    step KVBAttendWaitV _ _    _    _   _    _    _    vDv  _    _
      | vDv             = KVBAttendStep
      | otherwise       = KVBAttendWaitV
    step KVBAttendStep _  _    _    _   _    _    _    _    last' _
      | last'           = KVBAttendDone
      | otherwise       = KVBAttendK
    step KVBAttendDone _  _    _    _   _    _    _    _    _    _
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

    (out0, done0) = attentionHead clearSt stepEn dotAcc0 vRow isLastRow
    (out1, done1)
      | hasQ1    = attentionHead clearSt stepEn dotAcc1 vRow isLastRow
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
