{-# LANGUAGE DerivingVia #-}
module Simulation.DRAMBackedAxiSlave
  ( createDRAMBackedAxiSlave
  , createDRAMBackedAxiSlaveFromVec
  , WordData
  , DRAMConfig(..)
  ) where

import Clash.Prelude
import Data.ByteString.Lazy (ByteString)
import Data.Maybe (fromMaybe, isJust)

import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified Prelude as P

import qualified Data.Binary.Get as BG
import qualified Parser
import qualified Simulation.Parameters as PARAM
import LLaMa2.Types.ModelConfig
  ( NumLayers, ModelDimension, HeadDimension, HiddenDimension
  , NumQueryHeads, NumKeyValueHeads )
import LLaMa2.Memory.LayerAddressing (LayerSeg(..))
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Numeric.Types (Mantissa)

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
toRamIx = truncateB

-- ===============================================================
-- Top-level
-- ===============================================================

createDRAMBackedAxiSlave ::
  forall dom.
  HiddenClockResetEnable dom =>
  ByteString ->
  Master.AxiMasterOut dom ->
  Slave.AxiSlaveIn dom
createDRAMBackedAxiSlave modelBin = createDRAMBackedAxiSlaveFromVec defaultCfg initMem
  where
    defaultCfg = DRAMConfig { readLatency = 1, writeLatency = 0, numBanks = 1 }
    
    -- Parse the model using legacy parser
    params = BG.runGet Parser.parseLLaMa2ConfigFile modelBin
    
    -- Convert to address-indexed memory
    initMem = buildMemoryFromParams params

-- | Build a Vec 65536 WordData from parsed DecoderParameters
--   Maps hardware addresses to weight data in streaming format
buildMemoryFromParams :: PARAM.DecoderParameters -> Vec 65536 WordData
buildMemoryFromParams params = map addressToWord indicesI
  where
    -- Calculate address for a specific layer/segment/row
    -- Format: [layer][segment][head/row]
    -- Each layer: rmsAtt(1) + Q(nQ*hDim) + K(nKV*hDim) + V(nKV*hDim) + 
    --             WO(nQ*mDim) + rmsFfn(1) + W1(hDim) + W2(mDim) + W3(hDim)
    
    numLayers = natToNum @NumLayers
    modelDim = natToNum @ModelDimension
    headDim = natToNum @HeadDimension
    hiddenDim = natToNum @HiddenDimension
    numQHeads = natToNum @NumQueryHeads
    numKVHeads = natToNum @NumKeyValueHeads
    
    -- Rows per segment (matches rowsInSeg from LayerAddressing)
    rowsPerSeg :: LayerSeg -> Int
    rowsPerSeg seg = case seg of
      SegRmsAtt -> 1
      SegQ      -> numQHeads * headDim
      SegK      -> numKVHeads * headDim
      SegV      -> numKVHeads * headDim
      SegWO     -> numQHeads * modelDim
      SegRmsFfn -> 1
      SegW1     -> hiddenDim
      SegW2     -> modelDim
      SegW3     -> hiddenDim
    
    -- Total rows per layer
    rowsPerLayer = sum $ P.map rowsPerSeg [minBound .. maxBound]
    
    -- Bytes per row: (ModelDimension or HeadDimension or HiddenDimension) mantissas + 1 exponent
    -- For simplicity, assume max dimension rows are packed to 64-byte boundaries
    -- One RowI8E = Vec n (Signed 8) + Signed 8 exponent
    -- Pack into 64 bytes (one 512-bit word)
    
    addressToWord :: Index 65536 -> WordData
    addressToWord addr =
      let addrInt = fromEnum addr :: Int
          
          -- Determine which layer
          layerIdx = addrInt `div` rowsPerLayer
          rowInLayer = addrInt `mod` rowsPerLayer
          
      in if layerIdx >= numLayers
         then 0  -- Beyond valid layers
         else packRowToWord (getRowForLayerOffset layerIdx rowInLayer)

    getRowForLayerOffset :: Int -> Int -> RowI8E ModelDimension
    getRowForLayerOffset layerIdx offset =
      let layer = PARAM.modelLayers params !! layerIdx
          mha = PARAM.multiHeadAttention layer
          ffn = PARAM.feedforwardNetwork layer
          
          -- Segment boundaries within layer
          rmsAttEnd = 1
          qEnd = rmsAttEnd + numQHeads * headDim
          kEnd = qEnd + numKVHeads * headDim  
          vEnd = kEnd + numKVHeads * headDim
          woEnd = vEnd + numQHeads * modelDim
          rmsFfnEnd = woEnd + 1
          w1End = rmsFfnEnd + hiddenDim
          w2End = w1End + modelDim
          w3End = w2End + hiddenDim
          
      in if offset < rmsAttEnd
         then packRmsToRow (PARAM.rmsAttF mha)
         else if offset < qEnd
         then let qOffset = offset - rmsAttEnd
                  headIdx = qOffset `div` headDim
                  rowIdx = qOffset `mod` headDim
                  idx = fromIntegral headIdx :: Index NumQueryHeads
                  qMat = PARAM.wqHeadQ (PARAM.headsQ mha !! idx)
              in padRow (qMat !! rowIdx)
         else if offset < kEnd
         then let kOffset = offset - qEnd
                  headIdx = kOffset `div` headDim
                  rowIdx = kOffset `mod` headDim
                  queryHeadsPerKV = numQHeads `div` numKVHeads
                  qHeadIdx = headIdx * queryHeadsPerKV
                  kMat = PARAM.wkHeadQ (PARAM.headsQ mha !! qHeadIdx)
              in padRow (kMat !! rowIdx)
         else if offset < vEnd
         then let vOffset = offset - kEnd
                  headIdx = vOffset `div` headDim
                  rowIdx = vOffset `mod` headDim
                  queryHeadsPerKV = numQHeads `div` numKVHeads
                  qHeadIdx = headIdx * queryHeadsPerKV
                  vMat = PARAM.wvHeadQ (PARAM.headsQ mha !! qHeadIdx)
              in padRow (vMat !! rowIdx)
         else if offset < woEnd
         then let woOffset = offset - vEnd
                  headIdx = woOffset `div` modelDim
                  rowIdx = woOffset `mod` modelDim
                  woMat = PARAM.mWoQ mha !! headIdx
              in padRow (woMat !! rowIdx)
         else if offset < rmsFfnEnd
         then packRmsToRow (PARAM.fRMSFfnF ffn)
         else if offset < w1End
         then let rowIdx = offset - rmsFfnEnd
                  w1Mat = PARAM.fW1Q ffn
              in w1Mat !! rowIdx  -- Already ModelDimension wide
         else if offset < w2End
         then let rowIdx = offset - w1End
                  w2Mat = PARAM.fW2Q ffn
              in padRow (w2Mat !! rowIdx)
         else if offset < w3End
         then let rowIdx = offset - w2End
                  w3Mat = PARAM.fW3Q ffn
              in w3Mat !! rowIdx  -- Already ModelDimension wide
         else (repeat 0, 0)  -- Padding
    
    -- Pad a smaller row to ModelDimension by zero-filling
    -- Pad a smaller row to ModelDimension by zero-filling
    padRow :: forall n. KnownNat n => RowI8E n -> RowI8E ModelDimension
    padRow (mantissas, expon) = 
      let buildVec = imap (\i _ -> 
            if fromEnum i < natToNum @n 
            then mantissas !! (toEnum (fromEnum i) :: Index n)
            else 0) (repeat 0 :: Vec ModelDimension Mantissa)
      in (buildVec, expon)
      
    packRmsToRow :: Vec ModelDimension (SFixed 12 20) -> RowI8E ModelDimension
    packRmsToRow _weights =
      -- TODO: Implement proper quantization
      -- For now, return zeros (RMS weights handled separately in hardware)
      (repeat 0, 0)
    
    -- Pack a RowI8E into a 512-bit word (64 bytes)
    packRowToWord :: RowI8E ModelDimension -> WordData
    packRowToWord (mantissas, expon) =
      let -- Pack mantissas (ModelDimension signed 8-bit values)
          mantissaBits = foldl
            (\acc m -> (acc `shiftL` 8) .|. resize (pack m))
            (0 :: BitVector 512)
            mantissas
          -- Pack exponent in last byte
          expBits = resize (pack expon) :: BitVector 512
          -- Combine (exponent in LSB for simplicity)
      in (mantissaBits `shiftL` 8) .|. expBits

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
