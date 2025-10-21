module LLaMa2.Memory.AxiWriteMaster 
  (axiWriteMaster)
where

import Clash.Prelude
import LLaMa2.Memory.AXI
    ( AxiSlaveIn(bvalid, awready, wready),
      AxiMasterOut(..),
      AxiW(AxiW, wlast, wdata, wstrb),
      AxiAW(AxiAW, awid, awaddr, awlen, awsize, awburst) )

data WriteState
  = WriteIdle
  | WriteAddr     -- Send address
  | WriteData     -- Send data
  | WriteResp     -- Wait for response
  | WriteDone
  deriving (Generic, NFDataX, Show, Eq)

axiWriteMaster
  :: HiddenClockResetEnable dom
  => AxiSlaveIn dom
  -> Signal dom (Unsigned 32)             -- Write address
  -> Signal dom (BitVector 512)           -- Write data
  -> Signal dom Bool                      -- Start write
  -> ( AxiMasterOut dom
     , Signal dom Bool                    -- Write complete
     , Signal dom Bool                    -- Ready for new request
     )
axiWriteMaster slaveIn addrIn dataIn startWrite = (masterOut, writeDone, readyOut)
  where
    state = register WriteIdle nextState
    addr = regEn 0 startWrite addrIn
    writeData = regEn 0 startWrite dataIn
    
    nextState = mux (state .==. pure WriteIdle)
      (mux startWrite (pure WriteAddr) (pure WriteIdle))
      (mux (state .==. pure WriteAddr)
        (mux awHandshake (pure WriteData) (pure WriteAddr))
        (mux (state .==. pure WriteData)
          (mux wHandshake (pure WriteResp) (pure WriteData))
          (mux (state .==. pure WriteResp)
            (mux bHandshake (pure WriteDone) (pure WriteResp))
            (mux (state .==. pure WriteDone)
              (pure WriteIdle)
              (pure WriteIdle)))))
    
    -- AW channel
    awValid = state .==. pure WriteAddr
    awAddr = (\a -> AxiAW
      { awaddr  = a
      , awlen   = 0
      , awsize  = 6
      , awburst = 1
      , awid    = 0
      }) <$> addr
    awHandshake = awValid .&&. awready slaveIn
    
    -- W channel
    wValid = state .==. pure WriteData
    wData = (\d -> AxiW
      { wdata = d
      , wstrb = maxBound
      , wlast = True
      }) <$> writeData
    wHandshake = wValid .&&. wready slaveIn
    
    -- B channel
    bReady = state .==. pure WriteResp
    bHandshake = bReady .&&. bvalid slaveIn
    
    masterOut = AxiMasterOut
      { arvalid = pure False  -- Not reading
      , ardata  = pure (errorX "not reading")
      , rready  = pure False
      , awvalid = awValid
      , awdata  = awAddr
      , wvalid  = wValid
      , wdataMI = wData
      , bready  = bReady
      }
    
    writeDone = state .==. pure WriteDone
    readyOut = state .==. pure WriteIdle
