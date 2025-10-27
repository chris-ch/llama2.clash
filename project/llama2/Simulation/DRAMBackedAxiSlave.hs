module Simulation.DRAMBackedAxiSlave
  ( createDRAMBackedAxiSlave
  , WordData
  , DRAMConfig(..)
  ) where

import Clash.Prelude
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave  as Slave
import Data.Maybe (fromMaybe, isJust)

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

-- AXI4 arlen = beats-1. We keep "beatsLeft" as (arlen + 1).
beatsFromLen :: Unsigned 8 -> Unsigned 9
beatsFromLen l = resize l + 1

-- 512-bit data bus => 64 bytes per beat (INCR bursts).
incrAddr64B :: Unsigned 32 -> Unsigned 32
incrAddr64B a = a + 64

-- RAM address index: use low 16 bits → 64Ki entries of 512-bit words => 32MiB address space
toRamIx :: Unsigned 32 -> Unsigned 16
toRamIx = truncateB

-- ===============================================================
-- Top-level
-- ===============================================================

createDRAMBackedAxiSlave :: forall dom. HiddenClockResetEnable dom
  => DRAMConfig
  -> Vec 65536 WordData             -- initial contents
  -> Master.AxiMasterOut dom
  -> Slave.AxiSlaveIn dom
createDRAMBackedAxiSlave ramConfig initMem masterOut = slaveIn
 where
  
  writePathData :: WritePathData dom
  writePathData = writePath masterOut

  readPathData :: ReadPathData dom
  readPathData = readPath masterOut initMem ramConfig (writeOperation writePathData)

  -- ===============================================================
  -- Slave outputs
  -- ===============================================================
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
-- Read path (robust, per-beat controller)
-- ===============================================================

data ReadPathData dom = ReadPathData {
  addressReadReady :: Signal dom Bool,
  readValid :: Signal dom Bool,
  readData :: Signal dom AxiR
}

readPath :: forall dom.
  HiddenClockResetEnable dom
  =>  Master.AxiMasterOut dom
  -> Vec 65536 WordData
  -> DRAMConfig
  -> Signal dom (Maybe (Unsigned 16, WordData))
  -> ReadPathData dom
readPath masterOut initMem ramConfig ramOp = ReadPathData {
    addressReadReady = arReady,
    readValid = rValidReg,
    readData = rData
  }
  where
    -- Backing RAM (single-port, synchronous read)
    ram :: Signal dom (Unsigned 16)
        -> Signal dom (Maybe (Unsigned 16, WordData))
        -> Signal dom WordData
    ram = blockRamPow2 initMem

    -- AXI master inputs (read channels)
    arValid = Master.arvalid masterOut
    arData  = Master.ardata  masterOut
    rReady  = Master.rready  masterOut

    -- =======================
    -- Read channel state
    -- =======================
    rActive     = register False rActiveN
    rBeatsLeft  = register 0     rBeatsLeftN
    rAddr       = register 0     rAddrN
    rIDReg      = register 0     rIDRegN
    rIssuedAddr = register 0     rIssuedAddrN
    rWaitCnt    = register 0     rWaitCntN
    rValidReg   = register False rValidRegN
    rLastReg    = register False rLastRegN

    -- Handshakes
    arReady    = not <$> rActive
    arAccepted = arValid .&&. arReady
    rHandsh    = rValidReg .&&. rReady
    moreBeats  = (> 1) <$> rBeatsLeft
    launchBeat = arAccepted .||. (rHandsh .&&. moreBeats)

    -- Issue next read address
    nextIssueAddr =
      mux arAccepted (araddr <$> arData)
                     (incrAddr64B <$> rAddr)
    rIssuedAddrN = mux launchBeat nextIssueAddr rIssuedAddr

    -- Per-beat extra latency (BRAM contributes 1 implicitly)
    readLatU :: Unsigned 16
    readLatU = fromIntegral (max 0 (readLatency ramConfig))

    waiting = rWaitCnt ./=. pure 0
    rWaitCntN =
      mux launchBeat
        (pure readLatU)
      $ mux (waiting .&&. not <$> rValidReg)
        (rWaitCnt - 1)
        rWaitCnt

    -- Align rvalid to (BRAM 1 + readLatency)
    rValidRise = rActive .&&. (rWaitCnt .==. 1) .&&. not <$> rValidReg
    rValidRegN =
      mux rHandsh   (pure False) $
      mux rValidRise (pure True) rValidReg

    -- Bookkeeping
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

    -- =======================
    -- RAM and sticky RAW bypass
    -- =======================
    readIx  = toRamIx <$> rIssuedAddr
    ramData = ram readIx ramOp

    -- Capture last write (addr,data) on write handshake (ramOp=Just)
    wJust      = isJust <$> ramOp
    wIdxData   = fromMaybe (0, 0) <$> ramOp
    lastWIdx   = regEn 0 wJust (fst <$> wIdxData)
    lastWData  = regEn 0 wJust (snd <$> wIdxData)

    -- Sticky "pending-bypass" flag:
    --  - Set on any write handshake.
    --  - Cleared when we actually present a read beat (rvalid) to the same index
    --    and the master handshakes it (rready), i.e., when that forwarded data is consumed.
    pendingBypass :: Signal dom Bool
    pendingBypass = register False pendingBypassN

    hitForwardNow = pendingBypass .&&. (readIx .==. lastWIdx)
    consumeFwd    = hitForwardNow .&&. rHandsh  -- clear only when the beat is consumed

    pendingBypassN =
      mux wJust
        (pure True)
      $ mux consumeFwd
        (pure False)
        pendingBypass

    rPayload = mux hitForwardNow lastWData ramData

    -- R-channel payload
    rData = AxiR
          <$> rPayload
          <*> pure 0            -- RRESP = OKAY
          <*> rLastReg
          <*> rIDReg

-- ===============================================================
-- Write path (burst-capable, OKAY response)
-- ===============================================================

data WritePathData dom = WritePathData {
    addressWriteReady :: Signal dom Bool,       -- AWREADY
    writeReady  :: Signal dom Bool,             -- WREADY
    writeResponseValid  :: Signal dom Bool,     -- BVALID
    writeResponseData   :: Signal dom AxiB,     -- BRESP + BID
    writeOperation :: Signal dom (Maybe (Unsigned 16, WordData))
}

writePath :: forall dom.
  HiddenClockResetEnable dom
  =>  Master.AxiMasterOut dom
  -> WritePathData dom
writePath masterOut = WritePathData {
    addressWriteReady = awReady,
    writeReady        = wReadyS,
    writeResponseValid  = bValidReg,
    writeResponseData   = bData,
    writeOperation      = writeOp
 }
 where
    awValid = Master.awvalid masterOut
    awData  = Master.awdata  masterOut
    wValid  = Master.wvalid  masterOut
    wData   = Master.wdata   masterOut
    bReady  = Master.bready  masterOut

    -- State
    wActive    = register False wActiveN
    wBeatsLeft = register 0     wBeatsLeftN
    wAddrReg   = register 0     wAddrRegN
    wIDReg     = register 0     wIDRegN

    -- Handshakes
    awReady    = not <$> wActive
    awAccepted = awValid .&&. awReady

    -- Accept first W in the same cycle as AW
    wReadyS     = wActive .||. awAccepted
    wHandshSame = wReadyS .&&. wValid

    -- Pre/post values (atomic update)
    beatsOnAw     = beatsFromLen . awlen <$> awData
    preBeatsLeft  = mux awAccepted beatsOnAw wBeatsLeft
    postBeatsLeft = mux wHandshSame (preBeatsLeft - 1) preBeatsLeft

    preAddr  = mux awAccepted (awaddr <$> awData) wAddrReg
    postAddr = mux wHandshSame (incrAddr64B <$> preAddr) preAddr

    -- Register updates
    wBeatsLeftN = postBeatsLeft
    wAddrRegN   = postAddr
    wIDRegN     = mux awAccepted (awid <$> awData) wIDReg
    wActiveN    = postBeatsLeft ./=. 0

    -- Write operation: address before increment (this beat)
    writeIxThisBeat = toRamIx <$> preAddr
    wWord           = wdata <$> wData
    writeOp         = mux wHandshSame (Just <$> bundle (writeIxThisBeat, wWord))
                                     (pure Nothing)

    -- Single BRESP at end-of-burst; hold until BREADY
    bValidPulse = wHandshSame .&&. (postBeatsLeft .==. 0)
    bValidReg   = register False bValidRegN
    bValidRegN  =
      mux (bValidReg .&&. not <$> bReady) (pure True) $
      mux bValidPulse (pure True) (pure False)

    bData = AxiB 0 <$> wIDReg   -- BRESP=OKAY
