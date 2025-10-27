module Simulation.DRAMBackedAxiSlave
  ( createDRAMBackedAxiSlave
  , WordData
  , DRAMConfig(..)
  ) where

import Clash.Prelude
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave  as Slave

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

  readPathData = readPath masterOut initMem ramConfig (writeOperation writePathData)

  writePathData = writePath masterOut
  
  -- ===============================================================
  -- Slave outputs
  -- ===============================================================
  slaveIn = Slave.AxiSlaveIn
    { arready = arReady readPathData
    , rvalid  = rValid readPathData
    , rdata   = rData readPathData
    , awready = awready writePathData
    , wready  = wready writePathData
    , bvalid  = bvalid writePathData
    , bdata   = bdata writePathData
    }

-- ===============================================================
-- Read path (robust, per-beat controller)
-- ===============================================================

data ReadPathData dom = ReadPathData {
  arReady :: Signal dom Bool,
  rValid :: Signal dom Bool,
  rData :: Signal dom AxiR
}

readPath :: forall dom.
  HiddenClockResetEnable dom
  =>  Master.AxiMasterOut dom
  -> Vec 65536 WordData
  -> DRAMConfig
  -> Signal dom (Maybe (Unsigned 16, WordData))
  -> ReadPathData dom
readPath masterOut initMem ramConfig ramOp = ReadPathData {
    arReady = arReady,
    rValid = rValidReg,
    rData = rData
  }
  where

    -- ---------------------------------------------------------------
    -- Backing RAM (single port, synchronous read, write-enabled)
    -- ---------------------------------------------------------------
    ram :: Signal dom (Unsigned 16)
        -> Signal dom (Maybe (Unsigned 16, WordData))
        -> Signal dom WordData
    ram = blockRamPow2 initMem

    arValid = Master.arvalid masterOut
    arData  = Master.ardata  masterOut
    rReady  = Master.rready  masterOut
    
    -- State registers
    rActive    :: Signal dom Bool
    rActive    = register False rActiveN

    rBeatsLeft :: Signal dom (Unsigned 9) -- (arlen+1) .. 0
    rBeatsLeft = register 0 rBeatsLeftN

    rAddr      :: Signal dom (Unsigned 32) -- next-beat address
    rAddr      = register 0 rAddrN

    rIDReg     :: Signal dom (Unsigned 4)  -- widen to taste; uses arid width from your AxiAR
    rIDReg     = register 0 rIDRegN

    -- Address issued to RAM for the "currently pending" beat
    rIssuedAddr :: Signal dom (Unsigned 32)
    rIssuedAddr = register 0 rIssuedAddrN

    -- Simple per-beat countdown before rvalid (extra latency only).
    -- blockRamPow2 already adds 1 cycle for data; we trigger the RAM read
    -- at "launchBeat", then wait 'readLatency' cycles before asserting rvalid.
    rWaitCnt   :: Signal dom (Unsigned 16)
    rWaitCnt   = register 0 rWaitCntN

    -- rvalid is explicitly registered and cleared on handshake
    rValidReg  :: Signal dom Bool
    rValidReg  = register False rValidRegN

    -- rlast generated when the beat being presented is the last (beatsLeft == 1)
    rLastReg   :: Signal dom Bool
    rLastReg   = register False rLastRegN

    -- Handshake/accept lines
    arReady    :: Signal dom Bool
    arReady    = not <$> rActive

    arAccepted :: Signal dom Bool
    arAccepted = arValid .&&. arReady

    -- Launch a new beat:
    --  - when we just accepted AR (first beat), or
    --  - after an R handshake and more beats remain.
    rHandsh    :: Signal dom Bool
    rHandsh    = rValidReg .&&. rReady

    moreBeats  :: Signal dom Bool
    moreBeats  = (> 1) <$> rBeatsLeft

    launchBeat :: Signal dom Bool
    launchBeat = arAccepted .||. (rHandsh .&&. moreBeats)

    -- Next address to issue (at launch)
    nextIssueAddr :: Signal dom (Unsigned 32)
    nextIssueAddr =
      mux arAccepted
        (araddr <$> arData)
        (incrAddr64B <$> rAddr)

    -- Registers update
    rIssuedAddrN = mux launchBeat nextIssueAddr rIssuedAddr

    -- Countdown management: load on launch, count down while waiting for rvalid
    readLatU :: Unsigned 16
    readLatU = fromIntegral (max 0 (readLatency ramConfig))

    waiting  :: Signal dom Bool
    waiting  = rWaitCnt ./=. pure 0

    rWaitCntN =
      mux launchBeat
        (pure readLatU)
      $ mux (waiting .&&. not <$> rValidReg)
        (rWaitCnt - 1)
        rWaitCnt

    -- rvalid generation:
    -- - Assert when countdown reached 0 and we are active, and rvalid was not already asserted.
    -- - Deassert immediately after handshake.
    rValidRise :: Signal dom Bool
    rValidRise = rActive .&&. (rWaitCnt .==. 0) .&&. not <$> rValidReg

    rValidRegN =
      mux rHandsh
        (pure False)
      $ mux rValidRise
        (pure True)
        rValidReg

    -- Beats-left, address and active-state bookkeeping
    newBeatsVal :: Signal dom (Unsigned 9)
    newBeatsVal = beatsFromLen . arlen <$> arData

    rBeatsLeftN =
      mux arAccepted
        newBeatsVal
      $ mux rHandsh
        (rBeatsLeft - 1)
        rBeatsLeft

    rAddrN =
      mux arAccepted
        (araddr <$> arData)
      $ mux rHandsh
        (incrAddr64B <$> rAddr)
        rAddr

    rActiveN =
      mux arAccepted
        (pure True)
      $ mux (rHandsh .&&. (rBeatsLeft .==. 1))
        (pure False)
        rActive
    
    rIDRegN :: Signal dom (Unsigned 4)
    rIDRegN =
      mux arAccepted
        (arid <$> arData)
        rIDReg

    -- rlast is high exactly when we present the last beat
    rLastRegN =
      mux rValidReg
        (rBeatsLeft .==. 1)
        (pure False)

    -- Connect RAM: read index from the last issued address; write port defined below
    readIx :: Signal dom (Unsigned 16)
    readIx = toRamIx <$> rIssuedAddr

    ramData :: Signal dom WordData
    ramData = ram readIx ramOp

    -- R-channel payload
    rData :: Signal dom AxiR
    rData = AxiR
          <$> ramData
          <*> pure 0            -- RRESP = OKAY
          <*> rLastReg
          <*> rIDReg

-- ===============================================================
-- Write path (burst-capable, OKAY response)
-- ===============================================================

data WritePathData dom = WritePathData {
    awready :: Signal dom Bool,
    wready  :: Signal dom Bool,
    bvalid  :: Signal dom Bool,
    bdata   :: Signal dom AxiB,
    writeOperation :: Signal dom (Maybe (Unsigned 16, WordData))
}

writePath :: forall dom.
  HiddenClockResetEnable dom
  =>  Master.AxiMasterOut dom
  -> WritePathData dom
writePath masterOut = WritePathData {
    awready = awReady,
    wready  = wReadyS,
    bvalid  = bValidReg,
    bdata   = bData,
    writeOperation = writeOp
 }
 where

    awValid = Master.awvalid masterOut
    awData  = Master.awdata  masterOut
    wValid  = Master.wvalid  masterOut
    wData   = Master.wdata   masterOut
    bReady  = Master.bready  masterOut

    -- State
    wActive     :: Signal dom Bool
    wActive     = register False wActiveN

    wBeatsLeft  :: Signal dom (Unsigned 9)
    wBeatsLeft  = register 0 wBeatsLeftN

    wAddrReg    :: Signal dom (Unsigned 32)
    wAddrReg    = register 0 wAddrRegN

    wIDReg      :: Signal dom (Unsigned 4)
    wIDReg      = register 0 wIDRegN

    -- Handshakes
    awReady     :: Signal dom Bool
    awReady     = not <$> wActive

    awAccepted  :: Signal dom Bool
    awAccepted  = awValid .&&. awReady

    wReadyS     :: Signal dom Bool
    wReadyS     = wActive

    wHandsh     :: Signal dom Bool
    wHandsh     = wReadyS .&&. wValid

    -- bvalid (single response at end-of-burst)
    bValidReg   :: Signal dom Bool
    bValidReg   = register False bValidRegN

    -- Update logic
    wBeatsNew   :: Signal dom (Unsigned 9)
    wBeatsNew   = beatsFromLen . awlen <$> awData

    wActiveN =
      mux awAccepted
        (pure True)
      $ mux (wHandsh .&&. (wBeatsLeft .==. 1))
        (pure False)
        wActive

    wBeatsLeftN =
      mux awAccepted
        wBeatsNew
      $ mux wHandsh
        (wBeatsLeft - 1)
        wBeatsLeft

    wAddrRegN =
      mux awAccepted
        (awaddr <$> awData)
      $ mux wHandsh
        (incrAddr64B <$> wAddrReg)
        wAddrReg

    wIDRegN =
      mux awAccepted
        (awid <$> awData)
        wIDReg

    -- Generate a write operation on each accepted W beat
    wWord :: Signal dom WordData
    wWord = wdata <$> wData

    writeIx :: Signal dom (Unsigned 16)
    writeIx = toRamIx <$> wAddrReg

    writeOp :: Signal dom (Maybe (Unsigned 16, WordData))
    writeOp = mux wHandsh (Just <$> bundle (writeIx, wWord)) (pure Nothing)

    -- bvalid: pulse (and hold until bready) after the last W handshake
    bValidRegN =
      mux (bValidReg .&&. not <$> bReady)
        (pure True)
      $ mux (wHandsh .&&. (wBeatsLeft .==. 1))
        (pure True)
        (pure False)

    bData :: Signal dom AxiB
    bData = AxiB 0 <$> wIDReg   -- BRESP=OKAY
