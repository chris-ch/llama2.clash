module LLaMa2.Memory.WeightsLayout
  ( MatrixType(..)
  , WordsPerRow
  , rowAddressCalculator
  , kvRowAddressCalculator    -- NEW: Type-safe K/V address calculator
  , axiRowFetcher
  , axiMultiWordRowFetcher
  , multiWordRowParser
  , multiWordRowPacker
  , fixedPointVecPacker
  , matrixMultiWordPacker
  , rowStrideBytesI8E
  , align64
  , rowParser
  , requestCaptureStage
  , wordsPerRowVal
  , wordsPerFixedPointVec  -- Export for testing
  , FetcherDebug(..)           -- export debug type
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import LLaMa2.Numeric.Types ( Exponent, FixedPoint )
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut(..))
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn(..))
import qualified GHC.TypeNats as T
import Data.Type.Bool (If)
import qualified Prelude as P
import Clash.Sized.Vector (unsafeFromList)

-- | Calculate how many 64-byte words are needed for a RowI8E.
-- New layout:
--   - Word 0: 63 mantissas at bytes [0..62], exponent at byte 63
--   - Words 1..: 64 mantissas per word (no padding)
-- Therefore: 1 word if dim <= 63; else 1 + floor(dim/64).
type family WordsPerRow (dim :: Nat) :: Nat where
  WordsPerRow dim = If (dim <=? 63) 1 (1 + Div dim 64)

-- | Runtime value (must match the type-level WordsPerRow above)
wordsPerRowVal :: forall dim. KnownNat dim => Int
wordsPerRowVal =
  let d = natToNum @dim :: Int
  in if d <= 63 then 1 else 1 + (d `div` 64)

-- | Calculate words needed for a Vec n FixedPoint
-- Must match fixedPointVecPacker in DRAMBackedAxiSlave
wordsPerFixedPointVec :: forall n. KnownNat n => Int
wordsPerFixedPointVec =
  let n = natToNum @n :: Int
      -- FixedPoint is typically 32 bits = 4 bytes
      -- If you have a different size, adjust or use: natToNum @(BitSize FixedPoint `Div` 8)
      bytesPerEl = 4
      perWord = 64 `div` bytesPerEl  -- 16 FixedPoints per 64-byte word
  in (n + perWord - 1) `div` perWord

-- | Matrix type identifier for address calculation
data MatrixType = QMatrix | KMatrix | VMatrix | WOMatrix | W1Matrix | W2Matrix | W3Matrix
  deriving (Show, Eq, Generic, NFDataX)

data MultiWordFetchState (n :: Nat) = MWIdle | MWWaitAR | MWWaitR (Index n) | MWDone
  deriving (Show, Generic, NFDataX)

-- Equality requires some type manipulation
instance KnownNat n => Eq (MultiWordFetchState n) where
  (==) :: MultiWordFetchState n -> MultiWordFetchState n -> Bool
  MWIdle == MWIdle = True
  MWWaitAR == MWWaitAR = True
  (MWWaitR i) == (MWWaitR j) = i == j
  MWDone == MWDone = True
  _ == _ = False

-- | Calculate stride in bytes for a RowI8E matrix row
rowStrideBytesI8E :: forall n. KnownNat n => Int
rowStrideBytesI8E = wordsPerRowVal @n * 64

-- | Helper to align sizes to 64-byte boundaries
align64 :: Int -> Int
align64 n = ((n + 63) `div` 64) * 64

--------------------------------------------------------------------------------
-- Address Calculation - Internal Implementation
--------------------------------------------------------------------------------

-- | Internal address calculation that works with raw Int for head index.
-- This allows both Q-head and KV-head callers to use the same core logic.
rowAddressCalculatorInternal ::
  MatrixType
  -> Index NumLayers
  -> Int                    -- ^ Head index as Int (caller responsible for bounds)
  -> Index HeadDimension
  -> Unsigned 32
rowAddressCalculatorInternal matType layerIdx headIdxInt rowIdx =
  fromIntegral baseAddr + fromIntegral layerOffset +
  fromIntegral matrixOffset + fromIntegral headOffset + fromIntegral rowOffset
  where
    -- Model dimensions (compile-time constants)
    modelDim = natToNum @ModelDimension :: Int
    hiddenDim = natToNum @HiddenDimension :: Int
    numQHeads = natToNum @NumQueryHeads :: Int
    numKVHeads = natToNum @NumKeyValueHeads :: Int
    headDim = natToNum @HeadDimension :: Int
    vocabSize = natToNum @VocabularySize :: Int
    seqLen = natToNum @SequenceLength :: Int

    bytesPerWord = 64 :: Int

    -- Words per row for different data types
    wordsPerModelDimRowI8E = wordsPerRowVal @ModelDimension      -- RowI8E format
    wordsPerHiddenDimRowI8E = wordsPerRowVal @HiddenDimension    -- RowI8E format
    wordsPerModelDimFP = wordsPerFixedPointVec @ModelDimension   -- FixedPoint format
    wordsPerRotaryDimFP = wordsPerFixedPointVec @RotaryPositionalEmbeddingDimension

    -- ========== Global sections (before layers) ==========

    -- Embedding: vocabSize rows × ModelDimension each (RowI8E format)
    embeddingBytes = vocabSize * wordsPerModelDimRowI8E * bytesPerWord

    -- RMS Final: Vec ModelDimension FixedPoint (NOT RowI8E!)
    rmsFinalBytes = align64 (wordsPerModelDimFP * bytesPerWord)

    -- Rotary: 2 * seqLen rows of Vec RotaryDim FixedPoint (cos then sin)
    -- Each row is fixedPointVecPacker'd separately
    rotaryCosBytes = seqLen * wordsPerRotaryDimFP * bytesPerWord
    rotarySinBytes = seqLen * wordsPerRotaryDimFP * bytesPerWord
    rotaryRawBytes = rotaryCosBytes + rotarySinBytes
    rotaryBytes = align64 rotaryRawBytes

    baseAddr :: Int
    baseAddr = embeddingBytes + rmsFinalBytes + rotaryBytes

    -- ========== Per-layer sections ==========

    -- RMS Attention: Vec ModelDimension FixedPoint
    rmsAttBytes = align64 (wordsPerModelDimFP * bytesPerWord)

    -- Q matrices: NumQueryHeads × (HeadDim rows × ModelDim cols) - RowI8E
    qHeadBytes = headDim * wordsPerModelDimRowI8E * bytesPerWord
    qTotalBytes = numQHeads * qHeadBytes

    -- K/V matrices: NumKVHeads × (HeadDim rows × ModelDim cols) - RowI8E
    kHeadBytes = headDim * wordsPerModelDimRowI8E * bytesPerWord
    kTotalBytes = numKVHeads * kHeadBytes
    vTotalBytes = kTotalBytes

    -- WO matrix: NumQueryHeads × (ModelDim rows × HeadDim cols) - RowI8E
    woHeadBytes = modelDim * wordsPerRowVal @HeadDimension * bytesPerWord
    woTotalBytes = numQHeads * woHeadBytes

    -- RMS FFN: Vec ModelDimension FixedPoint
    rmsFfnBytes = align64 (wordsPerModelDimFP * bytesPerWord)

    -- W1: HiddenDim rows × ModelDim cols - RowI8E
    w1Bytes = hiddenDim * wordsPerModelDimRowI8E * bytesPerWord

    -- W2: ModelDim rows × HiddenDim cols - RowI8E
    w2Bytes = modelDim * wordsPerHiddenDimRowI8E * bytesPerWord

    -- W3: HiddenDim rows × ModelDim cols - RowI8E
    w3Bytes = hiddenDim * wordsPerModelDimRowI8E * bytesPerWord

    layerBytes = rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes +
                 rmsFfnBytes + w1Bytes + w2Bytes + w3Bytes

    layerOffset :: Int
    layerOffset = fromEnum layerIdx * layerBytes

    -- Matrix offset within layer (matches order in buildMemoryFromParams.layerWords)
    matrixOffset :: Int
    matrixOffset = case matType of
      QMatrix  -> rmsAttBytes
      KMatrix  -> rmsAttBytes + qTotalBytes
      VMatrix  -> rmsAttBytes + qTotalBytes + kTotalBytes
      WOMatrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes
      W1Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes + rmsFfnBytes
      W2Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes + rmsFfnBytes + w1Bytes
      W3Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes + rmsFfnBytes + w1Bytes + w2Bytes

    -- Head offset within matrix
    headBytes = case matType of
      QMatrix  -> qHeadBytes
      KMatrix  -> kHeadBytes
      VMatrix  -> kHeadBytes
      WOMatrix -> woHeadBytes
      _        -> 0  -- W1/W2/W3 don't have per-head indexing

    headOffset :: Int
    headOffset = headIdxInt * headBytes

    -- Row offset within head/matrix
    rowBytesForMatrix = case matType of
      W2Matrix -> wordsPerRowVal @HiddenDimension * bytesPerWord  -- W2 has HiddenDim columns
      WOMatrix -> wordsPerRowVal @HeadDimension   * bytesPerWord  -- WO has HeadDim columns
      _        -> wordsPerRowVal @ModelDimension * bytesPerWord   -- Q/K/V/W1/W3 have ModelDim columns

    rowOffset :: Int
    rowOffset = fromIntegral rowIdx * rowBytesForMatrix

--------------------------------------------------------------------------------
-- Public Address Calculators (Type-Safe Wrappers)
--------------------------------------------------------------------------------

-- | Calculate DDR byte address for Q, WO, W1, W2, W3 matrix rows.
-- Uses Index NumQueryHeads for head indexing (appropriate for Q and WO).
--
-- For K/V matrices, use 'kvRowAddressCalculator' instead.
--
-- CRITICAL: This must match the layout in buildMemoryFromParams exactly!
-- - RowI8E matrices use wordsPerRowVal (63 mantissas per word)
-- - FixedPoint vectors use wordsPerFixedPointVec (16 FP32 per word)
rowAddressCalculator ::
  MatrixType
  -> Index NumLayers
  -> Index NumQueryHeads      -- ^ Q/WO head index
  -> Index HeadDimension
  -> Unsigned 32
rowAddressCalculator matType layerIdx headIdx rowIdx =
  rowAddressCalculatorInternal matType layerIdx (fromEnum headIdx) rowIdx

-- | Calculate DDR byte address for K or V matrix rows.
-- Uses Index NumKeyValueHeads for head indexing (appropriate for K and V).
--
-- == Usage
--
-- @
-- kAddr = kvRowAddressCalculator KMatrix layerIdx kvHeadIdx rowIdx
-- vAddr = kvRowAddressCalculator VMatrix layerIdx kvHeadIdx rowIdx
-- @
--
-- == Type Safety
--
-- This function enforces that K/V matrices use the correct head index type,
-- preventing accidental use of Q head indices for K/V lookups (which would
-- cause address misalignment when NumQueryHeads /= NumKeyValueHeads).
--
kvRowAddressCalculator ::
  MatrixType
  -> Index NumLayers
  -> Index NumKeyValueHeads   -- ^ K/V head index (NOT NumQueryHeads!)
  -> Index HeadDimension
  -> Unsigned 32
kvRowAddressCalculator matType layerIdx kvHeadIdx rowIdx =
  rowAddressCalculatorInternal matType layerIdx (fromEnum kvHeadIdx) rowIdx

-- State machine for AXI read
data RowFetcherState = Idle | WaitAR | WaitR
    deriving (Show, Eq, Generic, NFDataX)

data CaptureState = CaptIdle | CaptRequesting | CaptProcessing
  deriving (Show, Eq, Generic, NFDataX)

-- | Request capture stage: 2-entry skid with combinational bypass.
-- Policies:
--   - requestAvail = frontValid || newRequest  (combinational)
--   - Do NOT enqueue when the consumer is ready and the queue is empty
--     (immediate accept: pulse is consumed, nothing stored).
--   - Pop/shift on the FALLING EDGE of consumerReady, so 'valid' remains
--     asserted across the cycle(s) where ready is high.
--   - If both entries are full and another pulse arrives, overwrite 'back'
--     with the newest request (depth-2 saturating behavior).
requestCaptureStage :: forall dom .
     HiddenClockResetEnable dom
  => Signal dom Bool                -- ^ newRequest (1-cycle pulse)
  -> Signal dom (Unsigned 32)       -- ^ newAddr
  -> Signal dom Bool                -- ^ consumerReady
  -> ( Signal dom Bool              -- ^ requestAvail (level)
     , Signal dom (Unsigned 32))    -- ^ capturedAddr (with combinational bypass)
requestCaptureStage newRequest newAddr consumerReady =
  (requestAvail, capturedAddr)
 where
  -- Queue state
  frontValid :: Signal dom Bool
  frontValid = register False frontValidN

  frontAddr  :: Signal dom (Unsigned 32)
  frontAddr  = register 0 frontAddrN

  backValid  :: Signal dom Bool
  backValid  = register False backValidN

  backAddr   :: Signal dom (Unsigned 32)
  backAddr   = register 0 backAddrN

  -- Combinational valid and bypassed address
  requestAvail :: Signal dom Bool
  requestAvail = frontValid .||. newRequest

  capturedAddr :: Signal dom (Unsigned 32)
  capturedAddr = mux newRequest newAddr frontAddr

  -- Ready falling-edge detection (pop policy)
  wasReady    :: Signal dom Bool
  wasReady     = register False consumerReady
  fallingEdge :: Signal dom Bool
  fallingEdge  = wasReady .&&. (not <$> consumerReady)

  -- Pop only on falling edge and only if a front item exists
  popFront :: Signal dom Bool
  popFront = fallingEdge .&&. frontValid

  -- State after potential pop
  frontValidAfterPop = mux popFront backValid frontValid
  frontAddrAfterPop  = mux popFront backAddr  frontAddr
  backValidAfterPop  = mux popFront (pure False) backValid
  backAddrAfterPop   = mux popFront (pure 0)     backAddr

  -- Immediate-accept detection: queue empty and consumer ready now
  queueEmptyAfterPop :: Signal dom Bool
  queueEmptyAfterPop = not <$> frontValidAfterPop

  immAccept :: Signal dom Bool
  immAccept = consumerReady .&&. queueEmptyAfterPop .&&. newRequest

  -- Only push when there is a pulse that is NOT immediately accepted
  willPush :: Signal dom Bool
  willPush = newRequest .&&. (not <$> immAccept)

  -- Push preference: front (if empty), else back (if empty), else overwrite back
  pushToFront :: Signal dom Bool
  pushToFront = willPush .&&. (not <$> frontValidAfterPop)

  pushToBack :: Signal dom Bool
  pushToBack  = willPush .&&.
                frontValidAfterPop .&&.
                (not <$> backValidAfterPop)

  overwriteBack :: Signal dom Bool
  overwriteBack = willPush .&&.
                  frontValidAfterPop .&&.
                  backValidAfterPop

  -- Next-state updates
  frontValidN = mux pushToFront (pure True)  frontValidAfterPop
  frontAddrN  = mux pushToFront newAddr      frontAddrAfterPop

  backValidN  = mux pushToBack  (pure True)
              $ mux overwriteBack (pure True)
              backValidAfterPop

  backAddrN   = mux pushToBack  newAddr
              $ mux overwriteBack newAddr
              backAddrAfterPop

-- | Pure AXI read state machine
axiRowFetcher :: forall dom.
     HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom              -- ^ From DRAM
  -> Signal dom Bool                   -- ^ Request pulse (1 cycle)
  -> Signal dom (Unsigned 32)          -- ^ Address to read
  -> ( Master.AxiMasterOut dom         -- ^ To DRAM
     , Signal dom (BitVector 512)      -- ^ Data (512 bits = 64 bytes)
     , Signal dom Bool                 -- ^ Data valid
     , Signal dom Bool)                -- ^ Ready (can accept new request)
axiRowFetcher slaveIn reqPulse addrIn =
  (masterOut, dataOut, dataValid, ready)
 where
  state = register SIdle nextState
  notInReset = register False (pure True)
  ready = (state .==. pure SIdle) .&&. notInReset
  start = ready .&&. reqPulse

  addrReg = regEn 0 start addrIn

  arvalidOut = state .==. pure SWaitAR
  ardataOut  = AxiAR <$> addrReg <*> pure 0 <*> pure 6 <*> pure 1 <*> pure 0
  rreadyOut  = pure True

  arAccepted = arvalidOut .&&. Slave.arready slaveIn
  rReceived  = Slave.rvalid slaveIn .&&. rreadyOut

  nextState =
    mux start (pure SWaitAR) $
    mux ((state .==. pure SWaitAR) .&&. arAccepted) (pure SWaitR) $
    mux ((state .==. pure SWaitR) .&&. rReceived) (pure SIdle) state

  rdataField = Slave.rdata slaveIn
  dataOut    = rdata <$> rdataField
  dataValid  = (state .==. pure SWaitR) .&&. Slave.rvalid slaveIn

  masterOut = Master.AxiMasterOut
    { arvalid = arvalidOut
    , ardata  = ardataOut
    , rready  = rreadyOut
    , awvalid = pure False
    , awdata  = pure (AxiAW 0 0 0 0 0)
    , wvalid  = pure False
    , wdata   = pure (AxiW 0 0 False)
    , bready  = pure False
    }
-- | Parse multiple 512-bit words into a RowI8E using the NEW layout:
-- Word 0: mant[0..62] at [0..62], exponent at byte 63.
-- Words 1..: 64 mantissas each (no per-word padding).
multiWordRowParser :: forall dim numWords.
  ( KnownNat dim
  , KnownNat (numWords T.* 64)
  , BitPack Exponent
  ) =>
  Vec numWords (BitVector 512) -> RowI8E dim
multiWordRowParser words' = RowI8E mantissas exponent'
  where
    -- Flatten bytes in beat order: w0 b0..b63, w1 b0..b63, ...
    allBytes :: Vec (numWords T.* 64) (BitVector 8)
    allBytes = concatMap (\w -> unpack w :: Vec 64 (BitVector 8)) words'

    -- Mantissa i -> byte i (i<63), else byte (i+1) (skip exponent slot at 63)
    mantBytes :: Vec dim (BitVector 8)
    mantBytes = imap
      (\i _ ->
         let iI = fromEnum i
             idx = if iI < 63 then iI else iI + 1
         in allBytes !! idx
      ) (repeat (0 :: Int))

    mantissas = map unpack mantBytes

    -- Exponent: byte 63 of the first word
    expByte  = allBytes !! (63::Int)

    exponent' :: Exponent
    exponent' =
      let e8 :: Signed 8
          e8 = unpack expByte
      in resize e8

-- Keep old single-word parser for backward compatibility
rowParser :: forall n. KnownNat n => BitVector 512 -> RowI8E n
rowParser word = multiWordRowParser (singleton word)


-- ============================================================================
-- Row packing (RowI8E -> 64B words)
-- ============================================================================

-- | Pack a RowI8E into multiple 64-byte words using the NEW layout:
--   Word 0: mant[0..62] at bytes 0..62, exponent at byte 63
--   Words 1..: 64 mantissas each (no per-word padding)
multiWordRowPacker :: forall dim. KnownNat dim => RowI8E dim -> [BitVector 512]
multiWordRowPacker row = go 0
  where
    dimI      = natToNum @dim :: Int
    numWords  = wordsPerRowVal @dim
    allMants  = toList $ rowMantissas row
    -- Exponent stored as full signed byte (two's complement).
    expByte   = pack (resize (rowExponent row) :: Signed 8) :: BitVector 8

    go :: Int -> [BitVector 512]
    go w
      | w >= numWords = []
      | w == 0        = packFirst : go (w+1)
      | otherwise     = packSubsequent w : go (w+1)

    -- Word 0: up to 63 mantissas, exponent at byte 63
    packFirst :: BitVector 512
    packFirst =
      let cnt0   = min 63 dimI
          m0     = P.take cnt0 allMants
          -- bytes 0..(cnt0-1): mantissas
          -- bytes cnt0..62   : zero padding
          mantBs = P.map pack m0 P.++ P.replicate (63 - cnt0) (0 :: BitVector 8)
          bytes  = mantBs P.++ [expByte]
      in pack (unsafeFromList (P.take 64 bytes) :: Vec 64 (BitVector 8))

    -- Word w>=1: 64 mantissas starting at index s = 63 + (w-1)*64
    packSubsequent :: Int -> BitVector 512
    packSubsequent w =
      let s      = 63 + (w - 1) * 64
          cnt    = max 0 (min 64 (dimI - s))
          mThis  = P.take cnt $ P.drop s allMants
          bytes  = P.map pack mThis P.++ P.replicate (64 - cnt) (0 :: BitVector 8)
      in pack (unsafeFromList (P.take 64 bytes) :: Vec 64 (BitVector 8))

-- Pack FixedPoint vector into 64-byte words (little-endian per element).
fixedPointVecPacker :: forall n. (KnownNat n, KnownNat (BitSize FixedPoint))
  => Vec n FixedPoint -> [BitVector 512]
fixedPointVecPacker v = go 0
  where
    nI         = natToNum @n :: Int
    bitSizeFP  = natToNum @(BitSize FixedPoint) :: Int
    bytesPerEl | bitSizeFP `mod` 8 /= 0 = error "FixedPoint BitSize must be a multiple of 8"
               | otherwise               = bitSizeFP `div` 8
    perWord    = 64 `div` bytesPerEl
    numWords   = (nI + perWord - 1) `div` perWord

    go w | w >= numWords = []
         | otherwise     = packWord w : go (w+1)

    packWord w =
      let s = w * perWord
          e = min (s + perWord) nI
          els = P.take (e - s) $ P.drop s $ toList v
          bytes = P.concatMap (\fp ->
                    let bits = pack fp :: BitVector (BitSize FixedPoint)
                    in [ resize (bits `shiftR` (8*i)) :: BitVector 8 | i <- [0 .. bytesPerEl-1] ]
                   ) els
          allB = bytes P.++ P.replicate (64 - P.length bytes) (0 :: BitVector 8)
      in pack (unsafeFromList (P.take 64 allB) :: Vec 64 (BitVector 8))

matrixMultiWordPacker :: forall rows cols. KnownNat cols
  => MatI8E rows cols -> [BitVector 512]
matrixMultiWordPacker m = P.concatMap multiWordRowPacker (toList m)

-- | Debug signals from the multi-word fetcher for assertion checking
data FetcherDebug dom = FetcherDebug
  { dbgLatchedAddr :: Signal dom (Unsigned 32)  -- Address actually latched by fetcher
  , dbgArAccepted  :: Signal dom Bool           -- AR handshake completed
  , dbgRReceived   :: Signal dom Bool           -- R data beat received
  , dbgBeat        :: Signal dom Int            -- Current beat counter as Int
  , dbgStateIsWaitR :: Signal dom Bool          -- State is MWWaitR
  , dbgStateIsDone :: Signal dom Bool           -- State is MWDone
  , dbgDoneCondition :: Signal dom Bool         -- The done transition condition
  }

-- Handshake-strict contract:
-- - Caller must assert requestPulse only when 'ready' is True.
-- - We latch the address synchronously on that handshake.
-- - No internal queueing, no combinational bypass, no reordering.
axiMultiWordRowFetcher :: forall dom dim.
     ( HiddenClockResetEnable dom
     , KnownNat (WordsPerRow dim)
     )
  => Slave.AxiSlaveIn dom
  -> Signal dom Bool                   -- ^ requestPulse (1-cycle) when ready==True
  -> Signal dom (Unsigned 32)          -- ^ base address
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec (WordsPerRow dim) (BitVector 512))
     , Signal dom Bool                 -- ^ dataValid (1-cycle pulse at completion)
     , Signal dom Bool                 -- ^ ready
     , FetcherDebug dom                -- ^ debug signals for assertions
     )
axiMultiWordRowFetcher slaveIn reqPulse addrIn =
  (masterOut, wordsOut, dataValid, ready, debugOut)
 where
  numWordsI = natToNum @(WordsPerRow dim) :: Int
  burstLen  = numWordsI - 1

  state :: Signal dom (MultiWordFetchState (WordsPerRow dim))
  state = register MWIdle nextState

  notInReset :: Signal dom Bool
  notInReset = register False (pure True)

  ready :: Signal dom Bool
  ready = (state .==. pure MWIdle) .&&. notInReset

  start :: Signal dom Bool
  start = ready .&&. reqPulse

  addrReg :: Signal dom (Unsigned 32)
  addrReg = regEn 0 start addrIn

  arvalidOut = state .==. pure MWWaitAR

  ardataOut = AxiAR
    <$> addrReg
    <*> pure (fromIntegral burstLen)  -- arlen
    <*> pure 6                        -- arsize = 64B
    <*> pure 1                        -- arburst INCR
    <*> pure 0                        -- arid

  rreadyOut = pure True

  arAccepted = arvalidOut .&&. Slave.arready slaveIn
  rReceived  = Slave.rvalid slaveIn .&&. rreadyOut

  -- beat counter
  beat :: Signal dom (Index (WordsPerRow dim))
  beat = register 0 nextBeat

  nextBeat =
    mux ((isWaitR <$> state) .&&. rReceived)
        (succWrap <$> beat)
        beat
   where
    isWaitR (MWWaitR _) = True
    isWaitR _           = False
    succWrap b = if b == maxBound then 0 else b + 1

  -- state transitions
  -- DEBUG: Check if we ever reach the done condition
  doneCondition = (state .==. pure (MWWaitR 0)) .&&. rReceived .&&. (beat .==. pure maxBound)
  
  nextState =
    mux start (pure MWWaitAR) $
    mux (arAccepted .&&. (state .==. pure MWWaitAR)) (pure (MWWaitR 0)) $
    mux doneCondition
         (pure MWDone) $
    mux (((\case
        MWWaitR _ -> True
        _ -> False) <$> state) .&&. rReceived)
         state
         (mux (state .==. pure MWDone) (pure MWIdle) state)

  -- Store beats
  rdataField :: Signal dom AxiR
  rdataField = Slave.rdata slaveIn

  currWord :: Signal dom (BitVector 512)
  currWord = rdata <$> rdataField

  bufInit = repeat 0 :: Vec (WordsPerRow dim) (BitVector 512)

  wordBuffer :: Signal dom (Vec (WordsPerRow dim) (BitVector 512))
  wordBuffer = register bufInit nextBuf

  nextBuf =
    mux (((\case
      MWWaitR _ -> True
      _ -> False) <$> state) .&&. rReceived)
        (replace <$> beat <*> currWord <*> wordBuffer)
        wordBuffer

  wordsOut  = wordBuffer
  dataValid = state .==. pure MWDone

  masterOut = Master.AxiMasterOut
    { arvalid = arvalidOut
    , ardata  = ardataOut
    , rready  = rreadyOut
    , awvalid = pure False
    , awdata  = pure (AxiAW 0 0 0 0 0)
    , wvalid  = pure False
    , wdata   = pure (AxiW 0 0 False)
    , bready  = pure False
    }

  -- Debug outputs for assertion checking
  isWaitRState = (\case MWWaitR _ -> True; _ -> False) <$> state
  isDoneState = (== MWDone) <$> state
  beatAsInt = fromEnum <$> beat  -- Convert Index to Int for debugging
  
  debugOut = FetcherDebug
    { dbgLatchedAddr = addrReg
    , dbgArAccepted  = arAccepted
    , dbgRReceived   = rReceived
    , dbgBeat        = beatAsInt
    , dbgStateIsWaitR = isWaitRState
    , dbgStateIsDone = isDoneState
    , dbgDoneCondition = doneCondition
    }

-- Single-beat version with the same strict handshake contract
data SState = SIdle | SWaitAR | SWaitR
  deriving (Generic, NFDataX, Show, Eq)
