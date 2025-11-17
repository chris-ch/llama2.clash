module LLaMa2.Memory.WeightStreaming
  ( MatrixType(..)
  , calculateRowAddress
  , axiRowFetcher
  , parseRow
  ) where

import Clash.Prelude
import qualified Prelude as P
import Clash.Sized.Vector (unsafeFromList)

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut(..))
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn(..))
import LLaMa2.Numeric.Types (Mantissa, Exponent)

-- | Matrix type identifier for address calculation
data MatrixType = QMatrix | KMatrix | VMatrix | WOMatrix | W1Matrix | W2Matrix | W3Matrix
  deriving (Show, Eq, Generic, NFDataX)

-- | Calculate DDR byte address for a specific matrix row
-- Example: calculateRowAddress 0 QMatrix 0 0 = 33345 (first Q byte)
calculateRowAddress ::
  MatrixType           -- ^ which matrix
  -> Index NumLayers      -- ^ which layer (0-based)
  -> Index NumQueryHeads  -- ^ head index (for Q/K/V)
  -> Index HeadDimension  -- ^ row index within matrix
  -> Unsigned 32          -- ^ output: DDR byte address
calculateRowAddress matType layerIdx headIdx rowIdx =
  fromIntegral baseAddr + fromIntegral layerOffset +
  fromIntegral matrixOffset + fromIntegral headOffset + fromIntegral rowOffset
  where
    -- Base address: after embedding, rmsFinal, and rotary sections
    vocabSize = natToNum @VocabularySize :: Int
    modelDim = natToNum @ModelDimension :: Int
    seqLen = natToNum @SequenceLength :: Int
    rotaryDim = natToNum @RotaryPositionalEmbeddingDimension :: Int

    embeddingBytes = vocabSize * (modelDim + 1)
    rmsFinalBytes = modelDim + 1
    rotaryBytes = 2 * seqLen * rotaryDim * 4

    baseAddr :: Int
    baseAddr = embeddingBytes + rmsFinalBytes + rotaryBytes

    -- Layer geometry
    numQHeads = natToNum @NumQueryHeads :: Int
    numKVHeads = natToNum @NumKeyValueHeads :: Int
    headDim = natToNum @HeadDimension :: Int
    hiddenDim = natToNum @HiddenDimension :: Int

    rmsAttBytes = modelDim + 1
    qHeadBytes = headDim * (modelDim + 1)
    qTotalBytes = numQHeads * qHeadBytes
    kHeadBytes = headDim * (modelDim + 1)
    kTotalBytes = numKVHeads * kHeadBytes
    vTotalBytes = kTotalBytes
    woBytes = modelDim * (numQHeads * headDim + 1)
    rmsFfnBytes = modelDim + 1
    w1Bytes = hiddenDim * (modelDim + 1)
    w2Bytes = modelDim * (hiddenDim + 1)
    w3Bytes = hiddenDim * (modelDim + 1)

    layerBytes = rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woBytes +
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
      W1Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woBytes + rmsFfnBytes
      W2Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woBytes + rmsFfnBytes + w1Bytes
      W3Matrix -> rmsAttBytes + qTotalBytes + kTotalBytes + vTotalBytes + woBytes + rmsFfnBytes + w1Bytes + w2Bytes

    -- Head offset within matrix
    headBytes = case matType of
      QMatrix -> qHeadBytes
      KMatrix -> kHeadBytes
      VMatrix -> kHeadBytes
      _ -> 0  -- WO, W1, W2, W3 don't have separate heads in storage

    headOffset :: Int
    headOffset = fromIntegral headIdx * headBytes

    -- Row offset within head/matrix
    rowBytes = case matType of
      W2Matrix -> hiddenDim + 1
      _ -> modelDim + 1

    rowOffset :: Int
    rowOffset = fromIntegral rowIdx * rowBytes

-- State machine for AXI read
data State = Idle | WaitAR | WaitR
    deriving (Show, Eq, Generic, NFDataX)

-- | Fetch one row (64 bytes) from DDR using AXI read protocol
-- Latency: ~2-3 cycles from request to valid data
axiRowFetcher ::
     HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom              -- ^ From DRAM
  -> Signal dom Bool             -- ^ Request pulse (fetch when True)
  -> Signal dom (Unsigned 32)    -- ^ Address to read
  -> ( Master.AxiMasterOut dom          -- ^ To DRAM
     , Signal dom (BitVector 512) -- ^ Data (512 bits = 64 bytes)
     , Signal dom Bool)           -- ^ Valid (data ready)
axiRowFetcher slaveIn request address = (masterOut, dataOut, validOut)
  where

    state = register Idle nextState

    -- Capture address when request arrives
    addressReg = regEn 0 request address

    -- AR channel (read address)
    arvalidOut = state .==. pure WaitAR
    ardataOut = AxiAR
      <$> addressReg
      <*> pure 0  -- arlen = 0 (single beat)
      <*> pure 6  -- arsize = 6 (2^6 = 64 bytes)
      <*> pure 1  -- arburst = 1 (INCR)
      <*> pure 0  -- arid = 0

    -- R channel (read data)
    rreadyOut = pure True  -- Always ready to receive

    -- Handshakes
    arAccepted = arvalidOut .&&. Slave.arready slaveIn
    rReceived = Slave.rvalid slaveIn .&&. rreadyOut

    -- State transitions
    nextState =
      mux (state .==. pure Idle .&&. request)
          (pure WaitAR)
      $ mux (state .==. pure WaitAR .&&. arAccepted)
          (pure WaitR)
      $ mux (state .==. pure WaitR .&&. rReceived)
          (pure Idle)
          state

    -- Extract data from AxiR
    rdataField = Slave.rdata slaveIn
    dataOut = rdata <$> rdataField
    validOut = (state .==. pure WaitR) .&&. Slave.rvalid slaveIn

    -- Construct master output (only read channels active)
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

-- | Parse a 512-bit word into RowI8E format
-- Extracts first n bytes as mantissas, byte n as exponent
-- For ModelDimension=64: bytes 0-62 are mantissas, byte 63 is exponent
-- (limited to 63 mantissas due to 64-byte word size)
parseRow :: forall n. KnownNat n => BitVector 512 -> RowI8E n
parseRow word = RowI8E {rowMantissas = mantissas, rowExponent = exponent'}
  where
    byteVec = unpack word :: Vec 64 (BitVector 8)
    bytes = toList byteVec :: [BitVector 8]

    n = natToNum @n :: Int

    -- Extract mantissas (up to 63, pad if needed)
    mantBytes = P.take n (bytes P.++ P.repeat 0)
    mantSigned = P.map (unpack :: BitVector 8 -> Mantissa) mantBytes
    mantissas = unsafeFromList mantSigned :: Vec n Mantissa

    -- Extract exponent (at byte n, or byte 63 if n >= 64)
    expIdx = min n 63
    expByte = bytes P.!! expIdx
    exponent' = unpack (resize expByte :: BitVector 7) :: Exponent
