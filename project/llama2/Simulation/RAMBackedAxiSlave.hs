module Simulation.RAMBackedAxiSlave
  ( createRAMBackedAxiSlave
  , ReadState(..)
  , WriteState(..)
  ) where
import Clash.Prelude
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Vector.Unboxed  as V
import Data.Word (Word8)
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut (..))
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn (..))
import LLaMa2.Memory.AXI.Types (AxiW (..), AxiB (..), AxiR (..), AxiAW (..), AxiAR (..))

-- ==================================================================
-- State definitions
-- ==================================================================
data ReadState  = RIdle | RBursting
  deriving (Generic, NFDataX, Show, Eq)

data WriteState = WIdle | WAccept | WBursting
  deriving (Generic, NFDataX, Show, Eq)

type WordData = BitVector 512

-- ==================================================================
-- Helper: build the initial Vec from a ByteString (run-time)
-- ==================================================================
initialVec :: BSL.ByteString -> Vec 65536 WordData
initialVec bs = foldr (\i v -> replace i (wordAt i) v) emptyVec (iterateI @65_536 (+1) 0)
  where
    bytes :: V.Vector Word8
    bytes = V.fromList (BSL.unpack bs)

    len = V.length bytes

    wordAt :: Int -> WordData
    wordAt idx =
      let base = idx * 64
          go :: Int -> BitVector 512 -> BitVector 512
          go i acc
            | i == 64   = acc
            | otherwise =
                let byteIdx = base + i
                    byteVal = if byteIdx < len
                              then fromIntegral (bytes V.! byteIdx) :: BitVector 8
                              else 0 :: BitVector 8
                in go (i + 1) (acc `shiftL` 8 .|. resize byteVal)
          in go 0 0

    emptyVec :: Vec 65536 WordData
    emptyVec = repeat 0

-- ==================================================================
-- Main AXI slave
-- ==================================================================

createRAMBackedAxiSlave
  :: forall dom . HiddenClockResetEnable dom
  => BSL.ByteString
  -> Master.AxiMasterOut dom
  -> (Slave.AxiSlaveIn dom, Signal dom ReadState, Signal dom WriteState)
createRAMBackedAxiSlave initFile masterOut = (slaveIn, readState, writeState)
 where
  ------------------------------------------------------------------
  -- BRAM initialised from the supplied ByteString
  ------------------------------------------------------------------
  initMem :: Vec 65536 WordData
  initMem = initialVec initFile

  ram
    :: Signal dom (Unsigned 16)                     -- read address
    -> Signal dom (Maybe (Unsigned 16, WordData))   -- write (addr,data)
    -> Signal dom WordData                          -- read data (next cycle)
  ram = blockRamPow2 initMem

  ------------------------------------------------------------------
  -- Register master inputs
  ------------------------------------------------------------------

  awvalid_r = register False (Master.awvalid masterOut)
  awdata_r  = register (AxiAW 0 0 0 0 0) (Master.awdata masterOut)
  awaddr_r  = awaddr <$> awdata_r
  wvalid_r  = register False (Master.wvalid masterOut)
  wdata_r   = register 0 (wdata <$> Master.wdata masterOut)
  wlast_r   = register False (wlast <$> (Master.wdata masterOut :: Signal dom AxiW))

  rready_r = register False (Master.rready masterOut)

  ------------------------------------------------------------------
  -- WRITE PATH (per-beat address increment)
  ------------------------------------------------------------------
  writeState = register WIdle nextWriteState
  awReady    = (== WIdle) <$> writeState
  awHandshake= awvalid_r .&&. awReady

  wReady     = (== WBursting) <$> writeState
  writeEn    = wvalid_r .&&. wReady

  -- Latch base write address on AW handshake
  latchedWriteAddr = regEn 0 awHandshake awaddr_r

  -- Per-burst beat counter (0..len)
  wBeatCnt :: Signal dom (Unsigned 8)
  wBeatCnt = register 0 nextWBeatCnt
  nextWBeatCnt =
    mux (writeState .==. pure WIdle) 0 $
    mux (writeState .==. pure WAccept) 0 $
    mux writeEn (wBeatCnt + 1) wBeatCnt

  -- Effective byte address = base + 64 * beatCnt
  writeWordAddr :: Signal dom (Unsigned 16)
  writeWordAddr =
    let offs = (* 64) . extend <$> wBeatCnt :: Signal dom (Unsigned 32)
    in resize . (`shiftR` 6) <$> (latchedWriteAddr + offs)

  writeOp :: Signal dom (Maybe (Unsigned 16, WordData))
  writeOp = mux writeEn (Just <$> bundle (writeWordAddr, wdata_r)) (pure Nothing)

  nextWriteState =
    mux (writeState .==. pure WIdle)
      (mux awHandshake (pure WAccept) (pure WIdle)) $
    mux (writeState .==. pure WAccept)
      (pure WBursting) $
    mux (writeEn .&&. wlast_r)
      (pure WIdle)
      (pure WBursting)

  lastWriteBeat = writeState .==. pure WBursting .&&. writeEn .&&. wlast_r

  -- proper BVALID retention
  bvalid_r = register False nextBValid
  nextBValid = (bvalid_r .&&. (not <$> Master.bready masterOut)) .||. lastWriteBeat
  bValid = bvalid_r

  bData  = pure (AxiB 0 0)   -- OKAY

  ------------------------------------------------------------------
  -- READ PATH
  ------------------------------------------------------------------
  readState = register RIdle nextReadState
  arReady = readState .==. pure RIdle

  arHandshake = Master.arvalid masterOut .&&. arReady

  currentArData = Master.ardata masterOut
  currentArLen = arlen <$> currentArData

  latchedReadAddr  = regEn 0 arHandshake (araddr <$> currentArData)
  latchedReadBeats = regEn 1 arHandshake (currentArLen + 1)  -- total beats

  -- raw signal that the slave is in bursting mode (pre-registered)
  rawBursting = readState .==. pure RBursting

  -- rValid is delayed one cycle to match BRAM read latency / readDataRaw
  rValid = register False rawBursting

  -- Beat counter increments only when the *output* data is accepted:
  -- i.e. when rValid && master.rready (we have registered rready earlier as rready_r)
  readBeatCounter :: Signal dom (Unsigned 8)
  readBeatCounter = register 0 nextReadBeatCounter
  nextReadBeatCounter =
    mux (readState .==. pure RIdle) 0 $
    mux (readState .==. pure RBursting)
      (mux (rValid .&&. rready_r) (readBeatCounter + 1) readBeatCounter)
      readBeatCounter

  -- compute read address (uses the unincremented beat counter: address for
  -- the beat being requested this cycle; BRAM returns it next cycle)
  readWordAddr :: Signal dom (Unsigned 16)
  readWordAddr =
    let offs = (* 64) . extend <$> readBeatCounter :: Signal dom (Unsigned 32)
    in resize . (`shiftR` 6) <$> (latchedReadAddr + offs)

  readDataRaw = ram readWordAddr writeOp

  -- rLast: should be true on the same cycle rValid is true when the data
  -- appearing on rdata is the last beat. To make that happen, compute the
  -- predicate one cycle earlier and register it.
  -- The data arriving when rValid==True corresponds to the beat index that
  -- was present in readBeatCounter *one cycle earlier*, so we check:
  --    (readBeatCounter + 1 >= latchedReadBeats)
  -- evaluated *before* the rValid cycle, and then register it.
  rLastNext = (readBeatCounter + 1 .>=. latchedReadBeats) .&&. rawBursting
  rLast = register False rLastNext

  rDataOut = AxiR <$> readDataRaw <*> pure 0 <*> rLast <*> pure 0

  -- A burst is done only when the last beat has been emitted AND accepted
  burstDone = rValid .&&. rready_r .&&. rLast

  nextReadState =
    mux (readState .==. pure RIdle)
        (mux arHandshake (pure RBursting) (pure RIdle)) $
    mux burstDone
        (pure RIdle)
        (pure RBursting)
  ------------------------------------------------------------------
  -- Output bundle
  ------------------------------------------------------------------
  slaveIn = Slave.AxiSlaveIn
    { arready = arReady
    , rvalid  = rValid
    , rdata = rDataOut
    , awready = awReady
    , wready  = wReady
    , bvalid  = bValid
    , bdata   = bData
    }
  