{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE AllowAmbiguousTypes #-}

module Simulation.DRAMBackedAxiSlave
  ( DRAMConfig(..)
  , WordData
  , createDRAMBackedAxiSlaveFromVec
  , createDRAMBackedAxiSlave
  , buildMemoryFromParams
  , wordsPerRowVal
  , packRowMultiWord
  ) where

import Clash.Prelude
import qualified Prelude as P

import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM
import Clash.Sized.Vector (unsafeFromList)

-- | Configuration for DRAM simulation timing
data DRAMConfig = DRAMConfig
  { arReadyDelay :: Int  -- Cycles before accepting AR
  , rValidDelay  :: Int  -- Cycles from AR accept to R valid
  , rBeatDelay   :: Int  -- Cycles between R beats in a burst
  } deriving (Show, Eq)

type WordData = BitVector 512  -- 64 bytes per word

-- ============================================================================
-- Compile-time dimension calculations
-- ============================================================================

-- | Calculate words per row at value level
wordsPerRowVal :: forall dim. KnownNat dim => Int
wordsPerRowVal =
  let dim = natToNum @dim :: Int
  in (dim + 62) `div` 63  -- Ceiling division: (dim + 1 exponent + 61 round) / 63
-- ===============================================================
-- Top-level
-- ===============================================================

createDRAMBackedAxiSlave ::
  forall dom.
  HiddenClockResetEnable dom =>
  PARAM.DecoderParameters ->
  Master.AxiMasterOut dom ->
  Slave.AxiSlaveIn dom
createDRAMBackedAxiSlave params = createDRAMBackedAxiSlaveFromVec defaultCfg initMem
  where
    defaultCfg = DRAMConfig { arReadyDelay = 1, rValidDelay = 1, rBeatDelay = 1 }

    -- Convert to address-indexed memory
    initMem = buildMemoryFromParams params

-- ============================================================================
-- Row Packing (Parameters -> DRAM)
-- ============================================================================

-- | Pack a single row into multiple 64-byte words
-- Layout per word:
--   Words 0..N-2: mantissas[i*63..(i+1)*63-1], byte 63 = 0 (padding)
--   Word N-1: mantissas[remaining], exponent, padding
packRowMultiWord :: forall dim. KnownNat dim => RowI8E dim -> [BitVector 512]
packRowMultiWord row = packWords 0
  where
    dim = natToNum @dim :: Int
    numWords = wordsPerRowVal @dim
    allMants = toList $ rowMantissas row
    exp' = rowExponent row

    packWords :: Int -> [BitVector 512]
    packWords wordIdx
      | wordIdx >= numWords = []
      | wordIdx == numWords - 1 = [packLastWord wordIdx]  -- Last word
      | otherwise = packMiddleWord wordIdx : packWords (wordIdx + 1)

    -- Middle words: 63 mantissas + 1 byte padding
    packMiddleWord :: Int -> BitVector 512
    packMiddleWord wordIdx =
      let startIdx = wordIdx * 63
          endIdx = min (startIdx + 63) dim
          mantsThisWord = P.take (endIdx - startIdx) $ P.drop startIdx allMants
          -- Pad to 64 bytes
          mantBytes = P.map pack mantsThisWord
          paddingCount = 64 - P.length mantBytes
          bytes = mantBytes P.++ P.replicate paddingCount (0 :: BitVector 8)
      in pack (unsafeFromList (P.take 64 bytes) :: Vec 64 (BitVector 8))

    -- Last word: remaining mantissas + exponent + padding
    packLastWord :: Int -> BitVector 512
    packLastWord wordIdx =
      let startIdx = wordIdx * 63
          numMantsInLast = dim - startIdx
          mantsThisWord = P.take numMantsInLast $ P.drop startIdx allMants
          expByte = resize (pack exp') :: BitVector 8
          mantBytes = P.map pack mantsThisWord
          -- Structure: [mantissas] [exponent] [padding]
          bytes = mantBytes
                  P.++ [expByte]
                  P.++ P.replicate (63 - numMantsInLast) (0 :: BitVector 8)
      in pack (unsafeFromList (P.take 64 bytes) :: Vec 64 (BitVector 8))

-- | Pack a matrix (collection of rows) into sequential words
packMatrixMultiWord :: forall rows cols. ( KnownNat cols)
  => MatI8E rows cols -> [BitVector 512]
packMatrixMultiWord matrix = P.concatMap packRowMultiWord (toList matrix)

-- | Pack FixedPoint vector into 64-byte aligned words (for RMS weights)
-- Each FixedPoint is 2 bytes (16 bits)
-- | Pack FixedPoint vector into 64-byte aligned words (for RMS weights)
-- Works for any BitSize FixedPoint that is a whole number of bytes.
packFixedPointVec :: forall n. (KnownNat n, KnownNat (BitSize FixedPoint))
  => Vec n FixedPoint -> [BitVector 512]
packFixedPointVec vec = packWords 0
  where
    n = natToNum @n :: Int
    bitSizeFP = natToNum @(BitSize FixedPoint) :: Int
    bytesPerElem =
      if bitSizeFP `mod` 8 /= 0
      then error "FixedPoint BitSize must be a whole multiple of 8"
      else bitSizeFP `div` 8
    elemsPerWord = 64 `div` bytesPerElem  -- number of FixedPoint elems per 64-byte word
    numWords = (n + elemsPerWord - 1) `div` elemsPerWord

    packWords :: Int -> [BitVector 512]
    packWords wordIdx
      | wordIdx >= numWords = []
      | otherwise = packWord wordIdx : packWords (wordIdx + 1)

    packWord :: Int -> BitVector 512
    packWord wordIdx =
      let startIdx = wordIdx * elemsPerWord
          endIdx = min (startIdx + elemsPerWord) n
          elemsThisWord = P.take (endIdx - startIdx) $ P.drop startIdx $ toList vec

          -- For each FixedPoint, pack it to its BitVector and then split into bytes (little-endian)
          bytes :: [BitVector 8]
          bytes = P.concatMap (\fp ->
                    let bits = pack fp :: BitVector (BitSize FixedPoint)
                        ind = [0 .. (bytesPerElem - 1)]
                    in P.map (\i -> resize (bits `shiftR` (8 * i)) :: BitVector 8) ind
                  ) elemsThisWord

          paddingCount = 64 - P.length bytes
          allBytes = bytes P.++ P.replicate paddingCount (0 :: BitVector 8)
      in pack (unsafeFromList (P.take 64 allBytes) :: Vec 64 (BitVector 8))

-- ============================================================================
-- Memory Building (Full Model -> DRAM)
-- ============================================================================

-- | Helper to align sizes to 64-byte boundaries
align64 :: Int -> Int
align64 n = ((n + 63) `div` 64) * 64

-- | Pad word list to 64-byte boundary
padTo64Bytes :: [BitVector 512] -> [BitVector 512]
padTo64Bytes wrds = wrds  -- Already 64-byte aligned since each word is 64 bytes

-- | Build complete DRAM memory from decoder parameters
buildMemoryFromParams :: PARAM.DecoderParameters -> Vec 65536 WordData
buildMemoryFromParams params =
  unsafeFromList $ P.take 65536 $ allWords P.++ P.repeat 0
  where
    -- Collect all sections in order
    allWords = embeddingWords
            P.++ rmsFinalWords
            P.++ rotaryWords
            P.++ P.concatMap layerWords [0..numLayers-1]

    numLayers = natToNum @NumLayers :: Int
    numQHeads = natToNum @NumQueryHeads :: Int
    numKVHeads = natToNum @NumKeyValueHeads :: Int

    embedding = PARAM.modelEmbedding params

    -- Embedding: vocabularyQ is MatI8E VocabularySize ModelDimension
    embeddingWords = packMatrixMultiWord (PARAM.vocabularyQ embedding)

    -- RMS Final: Vec ModelDimension FixedPoint
    rmsFinalWords = padTo64Bytes $ packFixedPointVec (PARAM.rmsFinalWeightF embedding)

    -- Rotary encoding tables: Vec SequenceLength (Vec RotaryDim FixedPoint)
    -- Need to pack both cos and sin tables
    rotaryWords =
      let
          -- Get rotary from first head of first layer
          layer0 = head (PARAM.modelLayers params)
          mha0 = PARAM.multiHeadAttention layer0
          head0 = head (PARAM.headsQ mha0)
          rotary = PARAM.rotaryF head0

          -- Pack cos table (SequenceLength × RotaryDim)
          cosTable = PARAM.freqCosF rotary
          cosWords = P.concatMap packFixedPointVec (toList cosTable)

          -- Pack sin table (SequenceLength × RotaryDim)
          sinTable = PARAM.freqSinF rotary
          sinWords = P.concatMap packFixedPointVec (toList sinTable)

          totalBytes = P.length cosWords * 64 + P.length sinWords * 64
          alignedBytes = align64 totalBytes
          paddingWords = (alignedBytes - totalBytes) `div` 64
      in cosWords P.++ sinWords P.++ P.replicate paddingWords 0

    -- Single layer's worth of weights
    layerWords :: Int -> [BitVector 512]
    layerWords layerIdx =
      let layer = PARAM.modelLayers params !! layerIdx
          mha = PARAM.multiHeadAttention layer
          ffn = PARAM.feedforwardNetwork layer
      in rmsAttWords mha
      P.++ qWords mha
      P.++ kWords mha
      P.++ vWords mha
      P.++ woWords mha
      P.++ rmsFfnWords ffn
      P.++ w1Words ffn
      P.++ w2Words ffn
      P.++ w3Words ffn

    -- RMS attention: Vec ModelDimension FixedPoint
    rmsAttWords mha = padTo64Bytes $ packFixedPointVec (PARAM.rmsAttF mha)

    -- Q matrices: all query heads
    -- headsQ :: Vec NumQueryHeads SingleHeadComponentQ
    -- wqHeadQ :: MatI8E HeadDimension ModelDimension
    qWords mha = P.concatMap qHeadWords [0..numQHeads-1]
      where
        qHeadWords headIdx =
          packMatrixMultiWord (PARAM.wqHeadQ (PARAM.headsQ mha !! headIdx))

    -- K matrices: need to map from KV heads
    -- For grouped-query attention: each KV head corresponds to multiple Q heads
    kWords mha = P.concatMap kHeadWords [0..numKVHeads-1]
      where
        queryHeadsPerKV = numQHeads `div` numKVHeads
        kHeadWords kvHeadIdx =
          let qHeadIdx = kvHeadIdx * queryHeadsPerKV
          in packMatrixMultiWord (PARAM.wkHeadQ (PARAM.headsQ mha !! qHeadIdx))

    -- V matrices: same mapping as K
    vWords mha = P.concatMap vHeadWords [0..numKVHeads-1]
      where
        queryHeadsPerKV = numQHeads `div` numKVHeads
        vHeadWords kvHeadIdx =
          let qHeadIdx = kvHeadIdx * queryHeadsPerKV
          in packMatrixMultiWord (PARAM.wvHeadQ (PARAM.headsQ mha !! qHeadIdx))

    -- WO matrices: mWoQ :: Vec NumQueryHeads (MatI8E ModelDimension HeadDimension)
    woWords mha = P.concatMap woHeadWords [0..numQHeads-1]
      where
        woHeadWords headIdx =
          packMatrixMultiWord (PARAM.mWoQ mha !! headIdx)

    -- RMS FFN: Vec ModelDimension FixedPoint
    rmsFfnWords ffn = padTo64Bytes $ packFixedPointVec (PARAM.fRMSFfnF ffn)

    -- W1: fW1Q :: MatI8E HiddenDimension ModelDimension
    w1Words ffn = packMatrixMultiWord (PARAM.fW1Q ffn)

    -- W2: fW2Q :: MatI8E ModelDimension HiddenDimension
    w2Words ffn = packMatrixMultiWord (PARAM.fW2Q ffn)

    -- W3: fW3Q :: MatI8E HiddenDimension ModelDimension
    w3Words ffn = packMatrixMultiWord (PARAM.fW3Q ffn)

-- ============================================================================
-- AXI Slave Implementation
-- ============================================================================

-- | State for AXI read transactions
data ReadState = RIdle | RProcessing (Index 256) (Index 256)
  deriving (Generic, NFDataX, Show)

instance Eq ReadState where
  RIdle == RIdle = True
  (RProcessing b1 c1) == (RProcessing b2 c2) = b1 == b2 && c1 == c2
  _ == _ = False

-- | Create an AXI slave backed by Vec memory with configurable timing
createDRAMBackedAxiSlaveFromVec :: forall dom.
  HiddenClockResetEnable dom
  => DRAMConfig
  -> Vec 65536 WordData
  -> Master.AxiMasterOut dom
  -> Slave.AxiSlaveIn dom
createDRAMBackedAxiSlaveFromVec config memory masterIn =
  Slave.AxiSlaveIn
    { arready = arreadySig
    , rvalid  = rvalidSig
    , rdata   = rdataSig
    , awready = pure False  -- Write not supported
    , wready  = pure False
    , bvalid  = pure False
    , bdata   = pure (AxiB 0 0)
    }
  where
    -- Read state machine
    readState :: Signal dom ReadState
    readState = register RIdle nextReadState

    -- Captured AR request
    capturedAR :: Signal dom AxiAR
    capturedAR = regEn (AxiAR 0 0 0 0 0) arAccepted (Master.ardata masterIn)

    -- Delay counters for timing simulation
    arDelayCounter :: Signal dom (Index 16)
    arDelayCounter = register 0 nextARDelay

    rDelayCounter :: Signal dom (Index 16)
    rDelayCounter = register 0 nextRDelay

    -- AR handshake signals
    arAccepted :: Signal dom Bool
    arAccepted = Master.arvalid masterIn .&&. arreadySig

    arreadySig :: Signal dom Bool
    arreadySig = (readState .==. pure RIdle) .&&. (arDelayCounter .==. pure 0)

    -- State transition
    nextReadState :: Signal dom ReadState
    nextReadState =
      mux arAccepted
          ((\ar -> RProcessing 0 (fromInteger $ toInteger $ arlen ar :: Index 256)) <$> Master.ardata masterIn)
          $ mux (isProcessing <$> readState .&&. rHandshake)
              (advance <$> readState)
              readState
      where
        isProcessing (RProcessing _ _) = True
        isProcessing RIdle             = False
        advance (RProcessing beat len)
          | beat >= len = RIdle
          | otherwise   = RProcessing (beat + 1) len
        advance s = s

    -- R channel handshake
    rHandshake :: Signal dom Bool
    rHandshake = rvalidSig .&&. Master.rready masterIn

    rvalidSig :: Signal dom Bool
    rvalidSig = (\s del -> case s of
                    RProcessing _ _ -> del == 0
                    _ -> False)
                <$> readState <*> rDelayCounter

    -- Extract current beat (0-based) from state
    currentBeat :: Signal dom (Index 256)
    currentBeat = (\case
        RProcessing b _ -> b
        RIdle -> 0) <$> readState

    -- Convert to Unsigned for address calculation
    beatOffset :: Signal dom (Unsigned 32)
    beatOffset = (`shiftL` 6) P.. fromIntegral P.<$> currentBeat

    -- Final read address
    currentAddress :: Signal dom (Unsigned 32)
    currentAddress = (+) <$> (araddr <$> capturedAR) <*> beatOffset

    -- Memory lookup
    memoryRead :: Signal dom WordData
    memoryRead = (\addr -> memory !! (addr `shiftR` 6)) <$> currentAddress

    -- R channel response
    rdataSig :: Signal dom AxiR
    rdataSig = (\s ar dat ->
                  let isLast = case s of
                                 RProcessing b len -> b >= len
                                 RIdle             -> False
                  in AxiR dat 0 isLast (arid ar)
               ) <$> readState <*> capturedAR <*> memoryRead

    -- Delay counter updates
    nextARDelay = mux arAccepted
                      (pure $ fromIntegral $ arReadyDelay config)
                  $ mux (arDelayCounter .>. pure 0)
                      (arDelayCounter - 1)
                      arDelayCounter

    nextRDelay = mux arAccepted
                     (pure $ fromIntegral $ rValidDelay config)
                 $ mux (rHandshake .&&. ((  \case
                      RProcessing _ _ -> True
                      _ -> False) <$> readState))
                     (pure $ fromIntegral $ rBeatDelay config)
                 $ mux (rDelayCounter .>. pure 0)
                     (rDelayCounter - 1)
                     rDelayCounter
