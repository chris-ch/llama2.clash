module LLaMa2.Memory.AxiReadMaster
  ( axiBurstReadMaster ) where

import Clash.Prelude
import LLaMa2.Memory.AXI
  ( AxiSlaveIn(..), AxiMasterOut(..), AxiR(..), AxiAR(..) )

data ReadState = ReadIdle | ReadAddr | ReadData | ReadDone
  deriving (Generic, NFDataX, Show, Eq)

axiBurstReadMaster :: HiddenClockResetEnable dom
  => AxiSlaveIn dom
  -> Signal dom (Unsigned 32)   -- ^ Base address
  -> Signal dom (Unsigned 8)    -- ^ Burst length (N transfers)
  -> Signal dom Bool            -- ^ Start read
  -> Signal dom Bool            -- ^ sinkReady (consumer ready for a beat)
  -> ( AxiMasterOut dom
     , Signal dom (BitVector 512) -- ^ Data stream
     , Signal dom Bool            -- ^ Data valid (beat)
     , Signal dom Bool            -- ^ Ready for new request
     )
axiBurstReadMaster slaveIn baseAddr burstLen startRead sinkReady =
  (masterOut, dataOut, validOut, readyOut)
 where
  state = register ReadIdle nextState
  addr  = regEn 0 startRead baseAddr
  len   = regEn 0 startRead burstLen
  cnt   = register (0 :: Unsigned 8) nextCnt

  isLast = cnt .==. len

  nextState = mux (state .==. pure ReadIdle)
                (mux startRead (pure ReadAddr) (pure ReadIdle))
             $ mux (state .==. pure ReadAddr)
                (mux arHS (pure ReadData) (pure ReadAddr))
             $ mux (state .==. pure ReadData)
                (mux (rHS .&&. isLast) (pure ReadDone) (pure ReadData))
             $ pure ReadIdle

  nextCnt = mux (state .==. pure ReadData .&&. rHS) (cnt + 1)
           $ mux (state .==. pure ReadIdle) 0 cnt

  arValid = state .==. pure ReadAddr
  arAddr  = (\a l -> AxiAR { araddr=a, arlen=l, arsize=6, arburst=1, arid=0 })
            <$> addr <*> len
  arHS    = arValid .&&. arready slaveIn

  rReady  = (state .==. pure ReadData) .&&. sinkReady
  rHS     = rReady .&&. rvalid slaveIn

  masterOut = AxiMasterOut
    { arvalid = arValid
    , ardata  = arAddr
    , rready  = rReady
    , awvalid = pure False
    , awdata  = pure (errorX "not writing")
    , wvalid  = pure False
    , wdataMI = pure (errorX "not writing")
    , bready  = pure False
    }

  dataOut  = rdata <$> rdataSI slaveIn
  validOut = rHS
  readyOut = state .==. pure ReadIdle
