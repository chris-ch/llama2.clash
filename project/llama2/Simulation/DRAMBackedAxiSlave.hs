{-# LANGUAGE DerivingVia #-}
module Simulation.DRAMBackedAxiSlave
  ( createDRAMBackedAxiSlave
  , createDRAMBackedAxiSlaveFromVec
  , WordData
  , DRAMConfig(..)
  ) where

import Clash.Prelude
import qualified Data.ByteString.Lazy as BSL
import Data.ByteString.Lazy (ByteString)
import Data.Maybe (fromMaybe, isJust)

import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified Prelude as P
import Data.Int (Int64)

-- ===============================================================
-- Configuration
-- ===============================================================

data DRAMConfig = DRAMConfig
  { readLatency  :: Int   -- extra per-beat latency cycles before rvalid
  , writeLatency :: Int   -- currently unused (writes are "posted")
  , numBanks     :: Int   -- currently unused (single-ported model)
  } deriving (Generic, NFDataX, Show, Eq)

type WordData = BitVector 512

-- ===============================================================
-- Helpers
-- ===============================================================

beatsFromLen :: Unsigned 8 -> Unsigned 9
beatsFromLen l = resize l + 1

incrAddr64B :: Unsigned 32 -> Unsigned 32
incrAddr64B a = a + 64

toRamIx :: Unsigned 32 -> Unsigned 16
toRamIx = truncateB

-- | Convert a ByteString into a Vec 65536 of 512-bit words (64 bytes each).
--   If the file is shorter than 64 * 65536 bytes (~4 MiB), the rest is zero-filled.
byteStringToVec512 :: ByteString -> Vec 65536 WordData
byteStringToVec512 bs = map getWord512 indicesI
  where
    totalBytes :: Int64
    totalBytes = BSL.length bs

    -- `i` selects the word index (0..65535). Result is a 512-bit word.
    getWord512 :: Index 65536 -> WordData
    getWord512 i =
      let base64 :: Int64
          base64 = fromIntegral (fromIntegral i :: Int) * 64

          -- getByte returns one byte at offset (base64 + j) as BitVector 8,
          -- zero if beyond `totalBytes`.
          getByte :: Int -> BitVector 8
          getByte j =
            let idx :: Int64
                idx = base64 + fromIntegral j
            in if idx < totalBytes
                 then fromIntegral (BSL.index bs idx) :: BitVector 8
                 else 0

          shiftAndMerge :: BitVector 512 -> Int -> BitVector 512
          shiftAndMerge acc j = (acc `shiftL` 8) .|. resize (getByte j)

          -- Assemble 64 bytes into a 512-bit WordData, MSB-first (big-endian).
          assemble :: WordData
          assemble = P.foldl shiftAndMerge 0 [0 .. 63]
      in assemble

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
    initMem = byteStringToVec512 modelBin

-- | Internal entry with explicit DRAMConfig and preloaded Vec.
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
