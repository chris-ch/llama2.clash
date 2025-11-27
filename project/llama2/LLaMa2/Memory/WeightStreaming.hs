module LLaMa2.Memory.WeightStreaming
  ( MatrixType(..)
  , calculateRowAddress
  , axiRowFetcher
  , axiMultiWordRowFetcher
  , multiWordRowParser
  , rowParser
  , requestCaptureStage
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut(..))
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn(..))
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import qualified GHC.TypeNats as T

-- | Calculate how many 64-byte words needed for a row
-- Each word can hold 63 mantissas (byte 63 reserved for last word's exponent)
-- Last word also holds the exponent
type family WordsPerRow (dim :: Nat) :: Nat where
  WordsPerRow dim = Div (dim + 62) 63  -- Ceiling division

-- | Helper to get the value at runtime
wordsPerRowVal :: forall dim. KnownNat dim => Int
wordsPerRowVal =
  let dim = natToNum @dim :: Int
  in (dim + 62) `div` 63

-- | Maximum burst length we might need (for largest model)
-- LLaMa 70B has ModelDimension = 7168
-- (7168 + 62) / 63 = 115 words
-- type MaxBurstWords = 128  -- Round up to power of 2 for safety

class KnownNat dim => RowHasEnoughBytes dim where
instance KnownNat dim => RowHasEnoughBytes dim  -- safe due to WordsPerRow definition

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

-- | Helper to align sizes to 64-byte boundaries
align64 :: Int -> Int
align64 n = ((n + 63) `div` 64) * 64

-- | Calculate DDR byte address for a specific matrix row
-- Enforces 64-byte alignment for all sections to match AXI word boundaries
calculateRowAddress ::
  MatrixType
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Index HeadDimension
  -> Unsigned 32
calculateRowAddress matType layerIdx headIdx rowIdx =
  fromIntegral baseAddr + fromIntegral layerOffset +
  fromIntegral matrixOffset + fromIntegral headOffset + fromIntegral rowOffset
  where
    modelDim = natToNum @ModelDimension :: Int
    hiddenDim = natToNum @HiddenDimension :: Int
    numQHeads = natToNum @NumQueryHeads :: Int
    numKVHeads = natToNum @NumKeyValueHeads :: Int
    headDim = natToNum @HeadDimension :: Int
    vocabSize = natToNum @VocabularySize :: Int
    seqLen = natToNum @SequenceLength :: Int
    rotaryDim = natToNum @RotaryPositionalEmbeddingDimension :: Int

    -- Calculate words per row for different matrix types
    wordsPerModelDimRow = wordsPerRowVal @ModelDimension
    wordsPerHiddenDimRow = wordsPerRowVal @HiddenDimension

    bytesPerWord = 64 :: Int

    -- Embedding: vocabSize rows × ModelDimension each
    embeddingBytes = vocabSize * wordsPerModelDimRow * bytesPerWord

    -- RMS Final: modelDim + 1 values (single row)
    rmsFinalWords = wordsPerRowVal @ModelDimension
    rmsFinalBytes = align64 (rmsFinalWords * bytesPerWord)

    -- Rotary: 2 * seqLen * rotaryDim * 4 bytes (float32)
    rotaryRaw = 2 * seqLen * rotaryDim * 4
    rotaryBytes = align64 rotaryRaw

    baseAddr :: Int
    baseAddr = embeddingBytes + rmsFinalBytes + rotaryBytes

    -- Layer geometry
    rmsAttBytes = align64 (rmsFinalWords * bytesPerWord)

    -- Q matrices: NumQueryHeads × (HeadDim rows × ModelDim cols)
    qHeadBytes = headDim * wordsPerModelDimRow * bytesPerWord
    qTotalBytes = numQHeads * qHeadBytes

    -- K/V matrices: NumKVHeads × (HeadDim rows × ModelDim cols)
    kHeadBytes = headDim * wordsPerModelDimRow * bytesPerWord
    kTotalBytes = numKVHeads * kHeadBytes
    vTotalBytes = kTotalBytes

    -- WO matrix: NumQueryHeads × (ModelDim rows × ModelDim cols)
    woTotalBytes = numQHeads * modelDim * wordsPerModelDimRow * bytesPerWord

    rmsFfnBytes = align64 (rmsFinalWords * bytesPerWord)

    -- W1/W3: HiddenDim rows × ModelDim cols
    w1Bytes = hiddenDim * wordsPerModelDimRow * bytesPerWord
    w3Bytes = hiddenDim * wordsPerModelDimRow * bytesPerWord

    -- W2: ModelDim rows × HiddenDim cols
    w2Bytes = modelDim * wordsPerHiddenDimRow * bytesPerWord

    layerBytes = rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes +
                 rmsFfnBytes + w1Bytes + w2Bytes + w3Bytes

    layerOffset :: Int
    layerOffset = fromEnum layerIdx * layerBytes

    -- Matrix offset within layer
    matrixOffset :: Int
    matrixOffset = case matType of
      QMatrix -> rmsAttBytes
      KMatrix -> rmsAttBytes + qTotalBytes
      VMatrix -> rmsAttBytes + qTotalBytes + kTotalBytes
      WOMatrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes
      W1Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes + rmsFfnBytes
      W2Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes + rmsFfnBytes + w1Bytes
      W3Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woTotalBytes + rmsFfnBytes + w1Bytes + w2Bytes

    -- Head offset within matrix
    headBytes = case matType of
      QMatrix -> qHeadBytes
      KMatrix -> kHeadBytes
      VMatrix -> kHeadBytes
      WOMatrix -> modelDim * wordsPerModelDimRow * bytesPerWord
      _ -> 0

    headOffset :: Int
    headOffset = fromIntegral headIdx * headBytes

    -- Row offset within head/matrix
    rowBytesForMatrix = case matType of
      W2Matrix -> wordsPerHiddenDimRow * bytesPerWord
      _ -> wordsPerModelDimRow * bytesPerWord

    rowOffset :: Int
    rowOffset = fromIntegral rowIdx * rowBytesForMatrix

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
axiRowFetcher :: forall dom .
     HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom              -- ^ From DRAM
  -> Signal dom Bool                   -- ^ Request pulse (1 cycle)
  -> Signal dom (Unsigned 32)          -- ^ Address to read
  -> ( Master.AxiMasterOut dom         -- ^ To DRAM
     , Signal dom (BitVector 512)      -- ^ Data (512 bits = 64 bytes)
     , Signal dom Bool                 -- ^ Data valid
     , Signal dom Bool)                -- ^ Ready (can accept new request)
axiRowFetcher slaveIn requestPulse address = (masterOut, dataOut, dataValid, ready)
  where
    -- *** STEP 1: Convert pulse to held request ***
    -- requestCapture holds the pulse until FSM can accept it
    (reqAvail, capturedAddr) = requestCaptureStage requestPulse address ready

    -- *** STEP 2: FSM state machine ***
    state :: Signal dom RowFetcherState
    state = register Idle nextState

    -- Track if we're out of reset (prevents accepting during reset)
    notInReset :: Signal dom Bool
    notInReset = register False (pure True)

    -- Capture address when starting a new transaction
    addressReg :: Signal dom (Unsigned 32)
    addressReg = regEn 0 startTransaction capturedAddr  -- Use capturedAddr from requestCapture

    -- Ready when idle AND out of reset
    ready :: Signal dom Bool
    ready = notInReset .&&. (state .==. pure Idle)

    -- Start new transaction when ready and request available
    startTransaction :: Signal dom Bool
    startTransaction = ready .&&. reqAvail  -- Use reqAvail from requestCapture

    -- AR channel (address read)
    arvalidOut :: Signal dom Bool
    arvalidOut = state .==. pure WaitAR

    ardataOut :: Signal dom AxiAR
    ardataOut = AxiAR
      <$> addressReg
      <*> pure 0  -- arlen = 0 (single beat)
      <*> pure 6  -- arsize = 6 (2^6 = 64 bytes)
      <*> pure 1  -- arburst = 1 (INCR)
      <*> pure 0  -- arid = 0

    -- R channel (read data)
    rreadyOut :: Signal dom Bool
    rreadyOut = pure True  -- Always ready to receive

    -- Handshakes
    arAccepted :: Signal dom Bool
    arAccepted = arvalidOut .&&. Slave.arready slaveIn

    rReceived :: Signal dom Bool
    rReceived = Slave.rvalid slaveIn .&&. rreadyOut

    -- State transitions
    nextState :: Signal dom RowFetcherState
    nextState =
      mux startTransaction
          (pure WaitAR)                                    -- Start new transaction
      $ mux (state .==. pure WaitAR .&&. arAccepted)
          (pure WaitR)                                     -- AR accepted, wait for data
      $ mux (state .==. pure WaitR .&&. rReceived)
          (pure Idle)                                      -- Data received, back to idle
          state                                            -- Stay in current state

    -- Output data
    rdataField :: Signal dom AxiR
    rdataField = Slave.rdata slaveIn

    dataOut :: Signal dom (BitVector 512)
    dataOut = rdata <$> rdataField

    dataValid :: Signal dom Bool
    dataValid = (state .==. pure WaitR) .&&. Slave.rvalid slaveIn

    -- Master output signals to DRAM
    masterOut :: Master.AxiMasterOut dom
    masterOut = Master.AxiMasterOut
      { arvalid = arvalidOut
      , ardata = ardataOut
      , rready = rreadyOut
      , awvalid = pure False
      , awdata = pure (AxiAW 0 0 0 0 0)
      , wvalid = pure False
      , wdata = pure (AxiW 0 0 False)
      , bready = pure False
      }

-- | Parse multiple 512-bit words into a RowI8E
-- Words are packed as: [mant0..mant62] [mant63..mant125] ... [mantN-1, exponent, padding]
multiWordRowParser :: forall dim numWords.
  ( KnownNat dim
  , KnownNat (numWords T.* 64)
  ) =>
  Vec numWords (BitVector 512) -> RowI8E dim
multiWordRowParser words' = RowI8E mantissas exponent'
  where
    -- Flatten all bytes
    allBytes :: Vec (numWords T.* 64) (BitVector 8)
    allBytes = concatMap (\w -> unpack w :: Vec 64 (BitVector 8)) words'

    dimI = natToNum @dim :: Int

    -- Manually index by Int, using (!!) which returns Maybe-free indexing
    mantBytes :: Vec dim (BitVector 8)
    mantBytes =
      imap (\i _ -> allBytes !! i) (repeat (0 :: Int))

    mantissas :: Vec dim Mantissa
    mantissas = map unpack mantBytes

    -- Exponent byte at index dim
    expByte :: BitVector 8
    expByte = allBytes !! dimI

    exponent' :: Exponent
    exponent' = unpack (resize expByte :: BitVector 7)

-- Keep old single-word parser for backward compatibility
rowParser :: forall n. KnownNat n => BitVector 512 -> RowI8E n
rowParser word = multiWordRowParser (singleton word)

-- | Generic multi-word AXI row fetcher
-- Fetches 'WordsPerRow dim' consecutive 64-byte words
axiMultiWordRowFetcher :: forall dom dim.
     (HiddenClockResetEnable dom, KnownNat (WordsPerRow dim))
  => Slave.AxiSlaveIn dom
  -> Signal dom Bool                   -- ^ Request pulse
  -> Signal dom (Unsigned 32)          -- ^ Base address
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec (WordsPerRow dim) (BitVector 512))  -- ^ All words
     , Signal dom Bool                 -- ^ Data valid
     , Signal dom Bool)                -- ^ Ready
axiMultiWordRowFetcher slaveIn requestPulse address =
  (masterOut, wordsOut, dataValid, ready)
  where
    numWords = natToNum @(WordsPerRow dim) :: Int
    burstLen = numWords - 1  -- AXI arlen is length - 1

    -- Request capture
    (reqAvail, capturedAddr) = requestCaptureStage requestPulse address ready

    -- State
    state :: Signal dom (MultiWordFetchState (WordsPerRow dim))
    state = register MWIdle nextState

    -- Word storage
    wordBuffer :: Signal dom (Vec (WordsPerRow dim) (BitVector 512))
    wordBuffer = register (repeat 0) nextWordBuffer

    -- Beat counter
    beatCounter :: Signal dom (Index (WordsPerRow dim))
    beatCounter = register 0 nextBeat

    notInReset :: Signal dom Bool
    notInReset = register False (pure True)

    addressReg :: Signal dom (Unsigned 32)
    addressReg = regEn 0 startTransaction capturedAddr

    ready :: Signal dom Bool
    ready = notInReset .&&. (state .==. pure MWIdle)

    startTransaction :: Signal dom Bool
    startTransaction = ready .&&. reqAvail

    -- AR channel
    arvalidOut :: Signal dom Bool
    arvalidOut = (\case
      MWWaitAR -> True
      _ -> False) <$> state

    ardataOut :: Signal dom AxiAR
    ardataOut = AxiAR
      <$> addressReg
      <*> pure (fromIntegral burstLen)  -- arlen = numWords - 1
      <*> pure 6                         -- arsize = 6 (64 bytes)
      <*> pure 1                         -- arburst = INCR
      <*> pure 0                         -- arid = 0

    -- R channel
    rreadyOut :: Signal dom Bool
    rreadyOut = pure True

    arAccepted :: Signal dom Bool
    arAccepted = arvalidOut .&&. Slave.arready slaveIn

    rReceived :: Signal dom Bool
    rReceived = Slave.rvalid slaveIn .&&. rreadyOut

    rdataField :: Signal dom AxiR
    rdataField = Slave.rdata slaveIn

    currentRData :: Signal dom (BitVector 512)
    currentRData = rdata <$> rdataField

    -- State transitions
    nextState :: Signal dom (MultiWordFetchState (WordsPerRow dim))
    nextState =
      mux startTransaction
          (pure MWWaitAR)
      $ mux ((\case
        MWWaitAR -> True
        _ -> False) <$> state .&&. arAccepted)
          (pure (MWWaitR 0))
      $ mux ((  \case
        MWWaitR _ -> True
        _ -> False
      ) <$> state .&&. rReceived)
          ((\s _ ->
            case s of
              MWWaitR n -> if n == maxBound then MWDone else MWWaitR (n + 1)
              _ -> s
           ) <$> state <*> beatCounter)
      $ mux ((  \case
        MWDone -> True
        _ -> False) <$> state)
          (pure MWIdle)
          state

    -- Update word buffer
    nextWordBuffer =
      mux ((  \case
        MWWaitR _ -> True
        _ -> False) <$> state .&&. rReceived)
          (replace <$> beatCounter <*> currentRData <*> wordBuffer)
          wordBuffer

    -- Update beat counter
    nextBeat =
      mux ((\case
        MWWaitR _ -> True
        _ -> False) <$> state .&&. rReceived)
          ((\beat -> if beat == maxBound then 0 else beat + 1) <$> beatCounter)
      $ mux ((\case
        MWDone -> True
        _ -> False) <$> state)
          (pure 0)
          beatCounter

    -- Outputs
    wordsOut = wordBuffer
    dataValid = (\case
      MWDone -> True
      _ -> False) <$> state

    masterOut = Master.AxiMasterOut
      { arvalid = arvalidOut
      , ardata = ardataOut
      , rready = rreadyOut
      , awvalid = pure False
      , awdata = pure (AxiAW 0 0 0 0 0)
      , wvalid = pure False
      , wdata = pure (AxiW 0 0 False)
      , bready = pure False
      }
