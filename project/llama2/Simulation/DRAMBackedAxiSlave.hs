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

createDRAMBackedAxiSlave :: forall dom. HiddenClockResetEnable dom
  => DRAMConfig
  -> Vec 65536 WordData             -- initial contents
  -> Master.AxiMasterOut dom
  -> Slave.AxiSlaveIn dom
createDRAMBackedAxiSlave ramConfig initMem masterOut = slaveIn
 where
  -- ---------------------------------------------------------------
  -- Backing RAM
  -- ---------------------------------------------------------------
  ram :: Signal dom (Unsigned 16)
      -> Signal dom (Maybe (Unsigned 16, WordData))
      -> Signal dom WordData
  ram = blockRamPow2 initMem

  readPathData :: ReadPath dom
  readPathData = readPath masterOut ramConfig ramData

  writePathData :: WritePathData dom
  writePathData = writePath masterOut

  -- SINGLE shared RAM instance with write-priority
  ramData :: Signal dom WordData
  ramData = ram (readAddress readPathData) (writeOperation writePathData)

  -- ---------------------------------------------------------------
  -- Slave Outputs
  -- ---------------------------------------------------------------
  slaveIn = Slave.AxiSlaveIn
    {
      arready = addressReadReady readPathData
    , rvalid  = readValid readPathData
    , rdata   = readData readPathData
    , awready = addressWriteReady writePathData
    , wready  = writeReady writePathData
    , bvalid  = writeResponseValid writePathData
    , bdata   = writeResponseData writePathData
    }

-- ---------------------------------------------------------------
-- Read Path
-- ---------------------------------------------------------------
data ReadPath dom = ReadPath {
  addressReadReady :: Signal dom Bool,
  readValid :: Signal dom Bool,
  readData :: Signal dom AxiR,
  readAddress :: Signal dom (Unsigned 16)
}

readPath :: forall dom . HiddenClockResetEnable dom
  => Master.AxiMasterOut dom -> DRAMConfig -> Signal dom WordData -> ReadPath dom
readPath masterOut ramConfig ramData = ReadPath {
    addressReadReady = readState .==. pure RIdle
    , readValid = rValid
    , readData = rData
    , readAddress = readAddr
  }
  where
    
    readState = register RIdle nextReadState

    -- Helper: check if currently in RBurst state
    isInRBurst :: Signal dom Bool
    isInRBurst = (  \case
      RBurst {} -> True
      _ -> False) <$> readState

    -- Burst completion: when counter reaches latchedLen during RBurst
    burstComplete :: Signal dom Bool
    burstComplete = isInRBurst .&&. (burstIdx .==. latchedLen) .&&. rValid .&&. rReady
    

    readBeatInc :: Signal dom Bool
    readBeatInc = isInRBurst .&&. rReady .&&. not <$> burstComplete

    burstIdx :: Signal dom (Unsigned 8)
    burstIdx = register 0 $ 
      mux burstComplete
          (pure 0)  -- Reset when burst completes
      $ mux readBeatInc 
          (burstIdx + 1) 
          burstIdx

    -- Next state logic
    nextReadState :: Signal dom ReadState
    nextReadState =
      mux (readState .==. pure RIdle .&&. arValid)
          (pure $ RBurst 0)
      $ mux burstComplete
          (pure RIdle)
          readState

    rReady  = Master.rready  masterOut
    arValid = Master.arvalid masterOut
    arData  = Master.ardata  masterOut

    -- Latch AR when idle and valid
    latchedAR :: Signal dom AxiAR
    latchedAR = regEn (AxiAR 0 0 0 0 0) enableLatch arData
      where
        enableLatch = arValid .&&. (readState .==. pure RIdle)

    latchedAddr :: Signal dom (Unsigned 32)
    latchedAddr = araddr <$> latchedAR

    latchedLen :: Signal dom (Unsigned 8)
    latchedLen  = arlen <$> latchedAR

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
        mux arAccepted (pure $ fromIntegral (readLatency ramConfig))
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

data WritePathData dom = WritePathData {
    addressWriteReady :: Signal dom Bool                     -- AWREADY
  , writeReady        :: Signal dom Bool                     -- WREADY
  , writeResponseValid:: Signal dom Bool                     -- BVALID
  , writeResponseData :: Signal dom AxiB                     -- BRESP + BID
  , writeOperation    :: Signal dom (Maybe (Unsigned 16, WordData))
  , writeAddress      :: Signal dom (Unsigned 16)
}

writePath :: forall dom. HiddenClockResetEnable dom
  => Master.AxiMasterOut dom -> WritePathData dom
writePath masterOut = WritePathData {
    addressWriteReady = writeState .==. pure WIdle
    , writeReady = isInWBurst .||. ((writeState .==. pure WIdle) .&&. awValid)
    , writeResponseValid = bValid
    , writeResponseData   = bData
    , writeOperation = writeOp
    , writeAddress = writeAddr
  } where
      awValid = Master.awvalid masterOut
      awData  = Master.awdata  masterOut
      wValid  = Master.wvalid  masterOut
      wData   = Master.wdata   masterOut
      bReady  = Master.bready  masterOut

      latchedAW = regEn (AxiAW 0 0 0 0 0) (awValid .&&. (writeState .==. pure WIdle)) awData
      latchedLast = wlast <$> wData

      writeState = register WIdle nextWriteState

      nextWriteState =
        mux (writeState .==. pure WResp .&&. bValid .&&. bReady)
            (pure WIdle) $
        mux ((writeState .==. pure WIdle) .&&. awValid .&&. wValid .&&. latchedLast)
            (pure WResp) $  -- Direct to WResp if everything arrives together
        mux ((writeState .==. pure WIdle) .&&. awValid)
            (pure (WBurst 0)) $
        mux (latchedLast .&&. wValid)
            (pure WResp)
            writeState

      bValid :: Signal dom Bool
      bValid = register False $ 
        (writeState .==. pure WResp) .&&. not <$> (bValid .&&. bReady)

      bData :: Signal dom AxiB
      bData = AxiB 0 <$> (awid <$> latchedAW)
      
      isInWBurst :: Signal dom Bool
      isInWBurst = (\case
        WBurst _ -> True
        _ -> False) <$> writeState

      latchedW  = wdata <$> wData
      
      -- Write address (truncated to 16-bit index)
      writeAddr :: Signal dom (Unsigned 16)
      writeAddr = truncateB . awaddr <$> latchedAW

      -- Write operation: (addr, data) when wValid, else Nothing
      writeOp :: Signal dom (Maybe (Unsigned 16, WordData))
      writeOp = mux wValid (Just <$> bundle (writeAddr, latchedW)) (pure Nothing)
