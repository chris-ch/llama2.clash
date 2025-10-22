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
  -> Signal dom (Unsigned 32)             
  -> Signal dom (BitVector 512)           
  -> Signal dom Bool                      
  -> ( AxiMasterOut dom
     , Signal dom Bool                    
     , Signal dom Bool                    
     )
axiWriteMaster slaveIn addrIn dataIn startWrite = (masterOut, writeDone, readyOut)
  where
    state = register WriteIdle nextState
    
    latchedAddr = regEn 0 startWrite addrIn
    latchedData = regEn 0 startWrite dataIn
    
    -- FORCE AW handshake completion (DEBUG)
    awValidRaw = state .==. pure WriteAddr
    awReady = awready slaveIn
    awHandshakeForced = awValidRaw .&&. awReady
    
    -- REGISTER EVERYTHING
    awAddr = regEn (errorX "aw uninit") awValidRaw 
           $ (\a -> AxiAW { awaddr=a, awlen=0, awsize=6, awburst=1, awid=0 }) <$> latchedAddr
    
    wData = regEn (errorX "w uninit") (state .==. pure WriteData)
          $ (\d -> AxiW { wdata=d, wstrb=maxBound, wlast=True }) <$> latchedData
    
    -- FIXED STATE MACHINE
    nextState = mux (state .==. pure WriteIdle)
      (mux startWrite (pure WriteAddr) (pure WriteIdle))
      (mux (state .==. pure WriteAddr)
        (mux awHandshakeForced           -- ✅ FORCE THIS!
          (pure WriteData) 
          (pure WriteAddr))
        (mux (state .==. pure WriteData)
          (mux (wValid .&&. wready slaveIn) 
            (pure WriteResp) 
            (pure WriteData))
          (mux (state .==. pure WriteResp)
            (mux (bReady .&&. bvalid slaveIn) 
              (pure WriteDone) 
              (pure WriteResp))
            (mux (state .==. pure WriteDone)
              (pure WriteIdle)
              (pure WriteIdle)))))

    awValid = state .==. pure WriteAddr
    wValid  = state .==. pure WriteData
    bReady  = state .==. pure WriteResp

    masterOut = AxiMasterOut
      { arvalid = pure False
      , ardata  = pure (errorX "no read")
      , rready  = pure False
      , awvalid = awValid
      , awdata  = awAddr
      , wvalid  = wValid
      , wdataMI = wData
      , bready  = bReady
      }

    writeDone = state .==. pure WriteDone
    readyOut  = state .==. pure WriteIdle
