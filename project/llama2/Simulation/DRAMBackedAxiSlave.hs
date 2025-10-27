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
  { readLatency  :: Int
  , writeLatency :: Int
  , numBanks     :: Int
  } deriving (Generic, NFDataX, Show, Eq)

type WordData = BitVector 512

data ReadState
  = RIdle
  | RBurst (Unsigned 8) -- burst counter (0 to len)
  deriving (Generic, NFDataX, Show, Eq)

data WriteState
  = WIdle
  | WBurst (Unsigned 8)
  | WResp
  deriving (Generic, NFDataX, Show, Eq)

-- ===============================================================
-- Top-level
-- ===============================================================

createDRAMBackedAxiSlave
  :: forall dom. HiddenClockResetEnable dom
  => DRAMConfig
  -> Vec 65536 WordData             -- initial contents
  -> Master.AxiMasterOut dom
  -> Slave.AxiSlaveIn dom
createDRAMBackedAxiSlave DRAMConfig{..} initMem masterOut = slaveIn
 where
  -- ---------------------------------------------------------------
  -- Backing RAM
  -- ---------------------------------------------------------------
  ram :: Signal dom (Unsigned 16)
      -> Signal dom (Maybe (Unsigned 16, WordData))
      -> Signal dom WordData
  ram = blockRamPow2 initMem

  -- ---------------------------------------------------------------
  -- Read Path
  -- ---------------------------------------------------------------
  readState = register RIdle nextReadState
  arValid = Master.arvalid masterOut
  arData  = Master.ardata  masterOut
  rReady  = Master.rready  masterOut

  -- Latch AR when idle and valid
  latchedAR :: Signal dom AxiAR
  latchedAR = regEn (AxiAR 0 0 0 0 0) enableLatch arData
    where
      enableLatch = arValid .&&. (readState .==. pure RIdle)

  latchedAddr :: Signal dom (Unsigned 32)
  latchedAddr = araddr <$> latchedAR

  latchedLen :: Signal dom (Unsigned 8)
  latchedLen  = arlen <$> latchedAR

  -- Helper: check if currently in RBurst state
  isInRBurst :: Signal dom Bool
  isInRBurst = (  \case
    RBurst {} -> True
    _ -> False) <$> readState

  -- Burst completion: when counter reaches latchedLen during RBurst
  burstComplete :: Signal dom Bool
  burstComplete = isInRBurst .&&. (burstIdx .==. latchedLen)

  -- Next state logic
  nextReadState :: Signal dom ReadState
  nextReadState =
    mux (readState .==. pure RIdle .&&. arValid)
        (pure $ RBurst 0)
    $ mux burstComplete
        (pure RIdle)
        readState

  readBeatInc :: Signal dom Bool
  readBeatInc = isInRBurst .&&. rReady

  burstIdx :: Signal dom (Unsigned 8)
  burstIdx = register 0 $ mux (readState .==. pure (RBurst 0)) 0 $ mux readBeatInc (burstIdx + 1) burstIdx

  burstIdx32 :: Signal dom (Unsigned 32)
  burstIdx32 = fromIntegral <$> burstIdx

  -- Byte offset = burstIdx * 64
  byteOffset :: Signal dom (Unsigned 32)
  byteOffset = (*) <$> burstIdx32 <*> pure 64
  -- or: byteOffset = (* 64) . fromIntegral <$> burstIdx

  -- Full 32-bit byte address
  fullAddr :: Signal dom (Unsigned 32)
  fullAddr = (+) <$> latchedAddr <*> byteOffset

  -- RAM index: lower 16 bits → 64 KiB * 512-bit = 32 MiB address space
  readAddr :: Signal dom (Unsigned 16)
  readAddr = truncateB <$> fullAddr

  arAccepted :: Signal dom Bool
  arAccepted = arValid .&&. (readState .==. pure RIdle)

  -- Counter for read latency
  rValidCounter :: Signal dom (Unsigned 16)
  rValidCounter = register 0 $
    mux arAccepted
        (pure $ fromIntegral readLatency)
    $ mux (rValidCounter .==. 0)
        (pure 0)
        (rValidCounter - 1)

  -- rValid high after latency, during burst
  rValid :: Signal dom Bool
  rValid = (rValidCounter .==. 0) .&&.
           (  \case
              RBurst {} -> True
              _ -> False) <$> readState

  -- rLast: when current beat is the last in burst
  rLast :: Signal dom Bool
  rLast = burstIdx .==. latchedLen

  -- Read response channel
  rData :: Signal dom AxiR
  rData = AxiR
        <$> ramData
        <*> pure 0           -- RRESP = OKAY
        <*> rLast
        <*> (arid <$> latchedAR)

  -- ---------------------------------------------------------------
  -- Write Path
  -- ---------------------------------------------------------------
  awValid = Master.awvalid masterOut
  awData  = Master.awdata  masterOut
  wValid  = Master.wvalid  masterOut
  wData   = Master.wdata   masterOut
  bReady  = Master.bready  masterOut

  latchedAW = regEn (AxiAW 0 0 0 0 0) (awValid .&&. (writeState .==. pure WIdle)) awData
  latchedW  = wdata <$> wData
  latchedLast = wlast <$> wData

  writeState = register WIdle nextWriteState

  nextWriteState =
    mux (writeState .==. pure WResp .&&. bValid .&&. bReady)
        (pure WIdle) $
    mux (writeState .==. pure WIdle)
      (mux awValid (pure (WBurst 0)) (pure WIdle)) $
    mux (latchedLast .&&. wValid)
      (pure WResp)
      writeState

  -- Write address (truncated to 16-bit index)
  writeAddr :: Signal dom (Unsigned 16)
  writeAddr = truncateB . awaddr <$> latchedAW

  -- Write operation: (addr, data) when wValid, else Nothing
  writeOp :: Signal dom (Maybe (Unsigned 16, WordData))
  writeOp = mux wValid (Just <$> bundle (writeAddr, latchedW)) (pure Nothing)

  -- SINGLE shared RAM instance with write-priority
  ramData :: Signal dom WordData
  ramData = ram readAddr writeOp
  
  bValid = register False $ (writeState .==. pure WResp) .||. (bValid .&&. not <$> bReady)
  bData = AxiB 0 <$> (awid <$> latchedAW)
  
  isInWBurst :: Signal dom Bool
  isInWBurst = (\case
    WBurst _ -> True
    _ -> False) <$> writeState

  -- ---------------------------------------------------------------
  -- Slave Outputs
  -- ---------------------------------------------------------------
  slaveIn = Slave.AxiSlaveIn
    { arready = readState .==. pure RIdle
    , rvalid  = rValid
    , rdata   = rData
    , awready = writeState .==. pure WIdle
    , wready  = isInWBurst  -- Accept write data only after address accepted
    , bvalid  = bValid
    , bdata   = bData
    }
