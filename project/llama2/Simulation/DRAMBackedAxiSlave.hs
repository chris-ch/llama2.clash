module Simulation.DRAMBackedAxiSlave
  ( createDRAMBackedAxiSlave
  , createDRAMBackedAxiSlaveFromVec
  , buildMemoryFromParams
  , packRowToWord
  , rowToBytes
  , layerToBytes
  , WordData
  , DRAMConfig(..)
  ) where

import Clash.Prelude
import Data.Maybe (fromMaybe, isJust)

import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified Prelude as P

import qualified Simulation.Parameters as PARAM
import LLaMa2.Types.ModelConfig
  ( ModelDimension
  , NumQueryHeads, NumKeyValueHeads, SequenceLength, RotaryPositionalEmbeddingDimension )
import LLaMa2.Numeric.Quantization (RowI8E (..))
import Simulation.Parameters (DecoderParameters)
import Clash.Sized.Vector (unsafeFromList)

-- ===============================================================
-- Configuration
-- ===============================================================

data DRAMConfig = DRAMConfig
  { readLatency  :: Int
  , writeLatency :: Int
  , numBanks     :: Int
  } deriving (Generic, NFDataX, Show, Eq)

type WordData = BitVector 512

-- ===============================================================
-- Smart DRAM that understands model structure
-- ===============================================================

beatsFromLen :: Unsigned 8 -> Unsigned 9
beatsFromLen l = resize l + 1

incrAddr64B :: Unsigned 32 -> Unsigned 32
incrAddr64B a = a + 64

toRamIx :: Unsigned 32 -> Unsigned 16
toRamIx addr = resize (addr `shiftR` 6)  -- Divide by 64 (shift right 6 bits)

-- ===============================================================
-- Top-level
-- ===============================================================

createDRAMBackedAxiSlave ::
  forall dom.
  HiddenClockResetEnable dom =>
  DecoderParameters ->
  Master.AxiMasterOut dom ->
  Slave.AxiSlaveIn dom
createDRAMBackedAxiSlave params = createDRAMBackedAxiSlaveFromVec defaultCfg initMem
  where
    defaultCfg = DRAMConfig { readLatency = 1, writeLatency = 0, numBanks = 1 }

    -- Convert to address-indexed memory
    initMem = buildMemoryFromParams params

buildMemoryFromParams :: PARAM.DecoderParameters -> Vec 65536 WordData
buildMemoryFromParams params = map wordAtAddress indicesI
  where
    -- Build complete file as flat byte array
    allBytes :: [BitVector 8]
    allBytes =
      embeddingBytes P.++
      rmsFinalBytes P.++
      rotaryBytes P.++
      layerBytes

    -- Extract 64 bytes starting at word index
    wordAtAddress :: Index 65536 -> WordData
    wordAtAddress wordIdx =
      let startByte = fromEnum wordIdx * 64
          slice' = P.take 64 $ P.drop startByte allBytes
          -- Pad if we run out of data
          padded = slice' P.++ P.replicate (64 - P.length slice') 0
          vecBytes = listToVecTH' padded :: Vec 64 (BitVector 8)
      in pack vecBytes

    -- EMBEDDING SECTION
    modelDim = natToNum @ModelDimension

    embeddingBytes =
      let vocab = PARAM.vocabularyQ (PARAM.modelEmbedding params)
      in P.concatMap rowToBytes (toList vocab)

    rmsFinalBytes = P.replicate (modelDim + 1) 0  -- RMS weights not quantized yet

    -- ROTARY SECTION (skip for now - not used in weight loading)
    seqLen = natToNum @SequenceLength
    rotaryDim = natToNum @RotaryPositionalEmbeddingDimension
    rotaryBytes = P.replicate (2 * seqLen * rotaryDim * 4) 0

    -- LAYER SECTION
    layerBytes = P.concatMap layerToBytes (toList (PARAM.modelLayers params))

-- Helper to convert list to Vec (use this or Template Haskell)
listToVecTH' :: forall n a. (KnownNat n, Default a) => [a] -> Vec n a
listToVecTH' xs =
  let len = natToNum @n
      padded = P.take len (xs P.++ P.repeat def)
  in unsafeFromList padded

-- Pack a RowI8E into a 512-bit word (64 bytes)
packRowToWord :: RowI8E ModelDimension -> WordData
packRowToWord RowI8E { rowMantissas = mantissas, rowExponent = expon} =
  let -- Take up to 63 mantissas (adjust if ModelDimension > 63)
      mantBytes = take (SNat @63) (map pack mantissas) :: Vec 63 (BitVector 8)
      expByte = resize (pack expon) :: BitVector 8

      -- Concatenate: [mant0, mant1, ..., mant62, exp]
      allBytes = mantBytes :< expByte :: Vec 64 (BitVector 8)

  in pack allBytes

-- Original Vec-based implementation for compatibility
createDRAMBackedAxiSlaveFromVec ::
  forall dom.
  HiddenClockResetEnable dom =>
  DRAMConfig ->
  Vec 65536 WordData ->
  Master.AxiMasterOut dom ->
  Slave.AxiSlaveIn dom
createDRAMBackedAxiSlaveFromVec ramConfig initMem masterOut = slaveIn
 where
  writePathData = writePath masterOut
  readPathData  = readPath masterOut initMem ramConfig (writeOperation writePathData)

  slaveIn = Slave.AxiSlaveIn
    { arready = addressReadReady readPathData
    , rvalid  = readValid readPathData
    , rdata   = readData readPathData
    , awready = addressWriteReady writePathData
    , wready  = writeReady writePathData
    , bvalid  = writeResponseValid writePathData
    , bdata   = writeResponseData writePathData
    }

-- ===============================================================
-- Read path
-- ===============================================================

data ReadPathData dom = ReadPathData
  { addressReadReady :: Signal dom Bool
  , readValid :: Signal dom Bool
  , readData :: Signal dom AxiR
  }

readPath ::
  forall dom.
  HiddenClockResetEnable dom =>
  Master.AxiMasterOut dom ->
  Vec 65536 WordData ->
  DRAMConfig ->
  Signal dom (Maybe (Unsigned 16, WordData)) ->
  ReadPathData dom
readPath masterOut initMem ramConfig ramOp = ReadPathData
  { addressReadReady = arReady
  , readValid = rValidReg
  , readData = rData
  }
  where
    ram :: Signal dom (Unsigned 16)
        -> Signal dom (Maybe (Unsigned 16, WordData))
        -> Signal dom WordData
    ram = blockRamPow2 initMem

    arValid = Master.arvalid masterOut
    arData  = Master.ardata  masterOut
    rReady  = Master.rready  masterOut

    rActive     = register False rActiveN
    rBeatsLeft  = register 0     rBeatsLeftN
    rAddr       = register 0     rAddrN
    rIDReg      = register 0     rIDRegN
    rIssuedAddr = register 0     rIssuedAddrN
    rWaitCnt    = register 0     rWaitCntN
    rValidReg   = register False rValidRegN
    rLastReg    = register False rLastRegN

    arReady    = not <$> rActive
    arAccepted = arValid .&&. arReady
    rHandsh    = rValidReg .&&. rReady
    moreBeats  = (> 1) <$> rBeatsLeft
    launchBeat = arAccepted .||. (rHandsh .&&. moreBeats)

    nextIssueAddr =
      mux arAccepted (araddr <$> arData)
                     (incrAddr64B <$> rAddr)
    rIssuedAddrN = mux launchBeat nextIssueAddr rIssuedAddr

    readLatU :: Unsigned 16
    readLatU = fromIntegral (max 0 (readLatency ramConfig))

    waiting = rWaitCnt ./=. pure 0
    rWaitCntN =
      mux launchBeat
        (pure readLatU)
      $ mux (waiting .&&. not <$> rValidReg)
        (rWaitCnt - 1)
        rWaitCnt

    rValidRise = rActive .&&. (rWaitCnt .==. 1) .&&. not <$> rValidReg
    rValidRegN =
      mux rHandsh   (pure False) $
      mux rValidRise (pure True) rValidReg

    newBeatsVal = beatsFromLen . arlen <$> arData
    rBeatsLeftN =
      mux arAccepted newBeatsVal $
      mux rHandsh   (rBeatsLeft - 1) rBeatsLeft

    rAddrN =
      mux arAccepted (araddr <$> arData) $
      mux rHandsh    (incrAddr64B <$> rAddr) rAddr

    rActiveN =
      mux arAccepted (pure True) $
      mux (rHandsh .&&. (rBeatsLeft .==. 1)) (pure False) rActive

    rIDRegN =
      mux arAccepted (arid <$> arData) rIDReg

    rLastRegN =
      mux rValidReg (rBeatsLeft .==. 1) (pure False)

    readIx  = toRamIx <$> rIssuedAddr
    ramData = ram readIx ramOp

    wJust      = isJust <$> ramOp
    wIdxData   = fromMaybe (0, 0) <$> ramOp
    lastWIdx   = regEn 0 wJust (fst <$> wIdxData)
    lastWData  = regEn 0 wJust (snd <$> wIdxData)

    pendingBypass = register False pendingBypassN
    hitForwardNow = pendingBypass .&&. (readIx .==. lastWIdx)
    consumeFwd    = hitForwardNow .&&. rHandsh

    pendingBypassN =
      mux wJust
        (pure True)
      $ mux consumeFwd
        (pure False)
        pendingBypass

    rPayload = mux hitForwardNow lastWData ramData

    rData = AxiR
          <$> rPayload
          <*> pure 0
          <*> rLastReg
          <*> rIDReg

-- ===============================================================
-- Write path
-- ===============================================================

data WritePathData dom = WritePathData
  { addressWriteReady :: Signal dom Bool
  , writeReady        :: Signal dom Bool
  , writeResponseValid :: Signal dom Bool
  , writeResponseData  :: Signal dom AxiB
  , writeOperation     :: Signal dom (Maybe (Unsigned 16, WordData))
  }

writePath ::
  forall dom.
  HiddenClockResetEnable dom =>
  Master.AxiMasterOut dom ->
  WritePathData dom
writePath masterOut = WritePathData
  { addressWriteReady = awReady
  , writeReady        = wReadyS
  , writeResponseValid  = bValidReg
  , writeResponseData   = bData
  , writeOperation      = writeOp
  }
  where
    awValid = Master.awvalid masterOut
    awData  = Master.awdata  masterOut
    wValid  = Master.wvalid  masterOut
    wData   = Master.wdata   masterOut
    bReady  = Master.bready  masterOut

    wActive    = register False wActiveN
    wBeatsLeft = register 0     wBeatsLeftN
    wAddrReg   = register 0     wAddrRegN
    wIDReg     = register 0     wIDRegN

    awReady    = not <$> wActive
    awAccepted = awValid .&&. awReady

    wReadyS     = wActive .||. awAccepted
    wHandshSame = wReadyS .&&. wValid

    beatsOnAw     = beatsFromLen . awlen <$> awData
    preBeatsLeft  = mux awAccepted beatsOnAw wBeatsLeft
    postBeatsLeft = mux wHandshSame (preBeatsLeft - 1) preBeatsLeft

    preAddr  = mux awAccepted (awaddr <$> awData) wAddrReg
    postAddr = mux wHandshSame (incrAddr64B <$> preAddr) preAddr

    wBeatsLeftN = postBeatsLeft
    wAddrRegN   = postAddr
    wIDRegN     = mux awAccepted (awid <$> awData) wIDReg
    wActiveN    = postBeatsLeft ./=. 0

    writeIxThisBeat = toRamIx <$> preAddr
    wWord           = wdata <$> wData
    writeOp         = mux wHandshSame (Just <$> bundle (writeIxThisBeat, wWord))
                                     (pure Nothing)

    bValidPulse = wHandshSame .&&. (postBeatsLeft .==. 0)
    bValidReg   = register False bValidRegN
    bValidRegN  =
      mux (bValidReg .&&. not <$> bReady) (pure True) $
      mux bValidPulse (pure True) (pure False)

    bData = AxiB 0 <$> wIDReg

-- Helper: convert one RowI8E to bytes
rowToBytes :: RowI8E n -> [BitVector 8]
rowToBytes RowI8E { rowMantissas = mantissas, rowExponent= expon} =
  let
    mantBytes :: [BitVector 8]
    mantBytes = P.map pack (toList mantissas)
    expByte :: BitVector 8
    expByte = resize (pack expon)
  in mantBytes P.++ [expByte]

layerToBytes :: PARAM.TransformerLayerComponent -> [BitVector 8]
layerToBytes layer =
  let mha = PARAM.multiHeadAttention layer
      ffn = PARAM.feedforwardNetwork layer

      modelDim = natToNum @ModelDimension

      -- MHA weight sections
      rmsAttBytes = P.replicate (modelDim + 1) 0  -- RMS not quantized

      qBytes mha' =
        let heads = toList (PARAM.headsQ mha')
        in P.concatMap (\hd ->
              let qMat = PARAM.wqHeadQ hd
              in P.concatMap rowToBytes (toList qMat)
            ) heads

      kBytes mha' =
        let kvHeadIndices = kvHeadIndicesFromQ
        in P.concatMap (\kvIdx ->
              let qHeadIdx = kvIdx * queryHeadsPerKV
                  kMat = PARAM.wkHeadQ (PARAM.headsQ mha' !! qHeadIdx)
              in P.concatMap rowToBytes (toList kMat)
            ) kvHeadIndices

      vBytes mha' =
        let kvHeadIndices = kvHeadIndicesFromQ
        in P.concatMap (\kvIdx ->
              let qHeadIdx = kvIdx * queryHeadsPerKV
                  vMat = PARAM.wvHeadQ (PARAM.headsQ mha' !! qHeadIdx)
              in P.concatMap rowToBytes (toList vMat)
            ) kvHeadIndices

      woBytes mha' =
        let heads = toList (PARAM.mWoQ mha')
        in P.concatMap (P.concatMap rowToBytes . toList) heads

      -- FFN weight sections  
      rmsFfnBytes = P.replicate (modelDim + 1) 0  -- RMS not quantized

      w1Bytes ffn' = P.concatMap rowToBytes (toList (PARAM.fW1Q ffn'))
      w2Bytes ffn' = P.concatMap rowToBytes (toList (PARAM.fW2Q ffn'))
      w3Bytes ffn' = P.concatMap rowToBytes (toList (PARAM.fW3Q ffn'))

      -- Helper values
      numKVHeads = natToNum @NumKeyValueHeads
      numQHeads = natToNum @NumQueryHeads
      queryHeadsPerKV = numQHeads `div` numKVHeads
      kvHeadIndicesFromQ = [(0 :: Int) .. numKVHeads - 1]

  in P.concat
    [ rmsAttBytes
    , qBytes mha
    , kBytes mha
    , vBytes mha
    , woBytes mha
    , rmsFfnBytes
    , w1Bytes ffn
    , w2Bytes ffn
    , w3Bytes ffn
    ]
