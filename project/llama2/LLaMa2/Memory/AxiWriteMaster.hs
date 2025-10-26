module LLaMa2.Memory.AxiWriteMaster
  ( axiWriteMaster )
where

import Clash.Prelude
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn (..))
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut (..))
import qualified LLaMa2.Memory.AXI.Types as AXITypes (AxiAW(..), AxiW (..))

-- Burst-capable write master:
-- - Latches base address and burst length on 'startBurst' (one-cycle pulse).
-- - Performs one AW handshake per burst, then streams 'len+1' beats on W.
-- - Asserts BREADY and completes on BVALID.
-- - Provides 'dataReady' so an upstream producer (e.g., read master) can backpressure us.
data WState = WIdle | WAddr | WData | WResp | WDone
  deriving (Generic, NFDataX, Eq, Show)

axiWriteMaster
  :: forall dom . HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Signal dom (Unsigned 32)   -- ^ base address (byte)
  -> Signal dom (Unsigned 8)    -- ^ burst length (N-1); e.g., 255 for 256 beats
  -> Signal dom Bool            -- ^ startBurst (one-cycle pulse)
  -> Signal dom (BitVector 512) -- ^ dataIn
  -> Signal dom Bool            -- ^ dataValid
  -> ( Master.AxiMasterOut dom
     , Signal dom Bool          -- ^ writeDone (one-cycle True at WDone)
     , Signal dom Bool          -- ^ dataReady (we can accept a beat now)
     )
axiWriteMaster slaveIn addrIn lenIn startBurst dataIn dataValid =
  ( masterOut
  , state .==. pure WDone
  , dataReady
  )
 where
  state = register WIdle nextState

  -- Latch addr/len at the start of a burst
  baseAddr = regEn 0 startBurst addrIn
  burstLen = regEn 0 startBurst lenIn

  -- Beat counter counts successful W handshakes (0 .. len)
  beatCnt :: Signal dom (Unsigned 8)
  beatCnt = register 0 nextBeatCnt

  isLastBeat = beatCnt .==. burstLen

  -- Channel signals
  awValid = state .==. pure WAddr
  awData  = AXITypes.AxiAW <$> baseAddr <*> burstLen <*> pure 6 <*> pure 1 <*> pure 0
  awHS    = awValid .&&. Slave.awready slaveIn

  wFire   = wValid .&&. Slave.wready slaveIn
  wValid  = (state .==. pure WData) .&&. dataValid
  wLast   = isLastBeat
  wData   = AXITypes.AxiW <$> dataIn <*> pure maxBound <*> wLast

  bReady  = state .==. pure WResp
  bHS     = bReady .&&. Slave.bvalid slaveIn

  -- Upstream backpressure: can accept a beat iff in data phase and slave is ready
  dataReady = (state .==. pure WData) .&&. Slave.wready slaveIn

  nextBeatCnt =
    mux (state .==. pure WIdle) 0 $
    mux (state .==. pure WAddr .&&. awHS) 0 $
    mux (state .==. pure WData .&&. wFire)
        (beatCnt + 1)
        beatCnt

  nextState =
    mux (state .==. pure WIdle)
      (mux startBurst (pure WAddr) (pure WIdle)) $
    mux (state .==. pure WAddr)
      (mux awHS (pure WData) (pure WAddr)) $
    mux (state .==. pure WData)
      (mux (wFire .&&. isLastBeat) (pure WResp) (pure WData)) $
    mux (state .==. pure WResp)
      (mux bHS (pure WDone) (pure WResp)) $
    -- WDone -> WIdle (one-shot done pulse)
    pure WIdle

  masterOut = Master.AxiMasterOut
    { arvalid = pure False
    , ardata  = pure (errorX "no read")
    , rready  = pure False
    , awvalid = awValid
    , awdata  = awData
    , wvalid  = wValid
    , wdata = wData
    , bready  = bReady
    }
  