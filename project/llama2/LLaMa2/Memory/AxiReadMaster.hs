module LLaMa2.Memory.AxiReadMaster 
  (axiBurstReadMaster)
where
import Clash.Prelude
import LLaMa2.Memory.AXI
    ( AxiSlaveIn(rdataSI, arready, rvalid),
      AxiMasterOut(..),
      AxiR(rdata),
      AxiAR(AxiAR, arid, araddr, arlen, arsize, arburst) )

-- Simple states for read transaction
data ReadState 
  = ReadIdle
  | ReadAddr      -- Sending address
  | ReadData      -- Receiving data
  | ReadDone
  deriving (Generic, NFDataX, Show, Eq)

-- Read multiple consecutive addresses (burst)
axiBurstReadMaster :: HiddenClockResetEnable dom
  => AxiSlaveIn dom
  -> Signal dom (Unsigned 32)             -- Base address
  -> Signal dom (Unsigned 8)              -- Burst length (N transfers)
  -> Signal dom Bool                      -- Start burst read
  -> ( AxiMasterOut dom
     , Signal dom (BitVector 512)         -- Data stream
     , Signal dom Bool                    -- Data valid (one per transfer)
     , Signal dom Bool                    -- Ready for new request
     )
axiBurstReadMaster slaveIn baseAddr burstLen startRead = 
  (masterOut, dataOut, validOut, readyOut)
  where
    state = register ReadIdle nextState
    
    -- ✅ CRITICAL: Latch burstLen ONCE per burst
    latchedBaseAddr = regEn 0 startRead baseAddr
    latchedBurstLen = regEn 0 startRead burstLen  -- FIXED!
    
    addr = latchedBaseAddr
    len = latchedBurstLen
    
    counter = register (0 :: Unsigned 8) nextCounter
    isLastTransfer = counter .==. len
    
    nextState = mux (state .==. pure ReadIdle)
      (mux startRead (pure ReadAddr) (pure ReadIdle))
      (mux (state .==. pure ReadAddr)
        (mux arHandshake (pure ReadData) (pure ReadAddr))
        (mux (state .==. pure ReadData)
          (mux (rHandshake .&&. isLastTransfer) 
            (pure ReadDone) 
            (pure ReadData))
          (mux (state .==. pure ReadDone)
            (pure ReadIdle)
            (pure ReadIdle))))
    
    nextCounter = mux (state .==. pure ReadData .&&. rHandshake)
      (counter + 1)
      (mux (state .==. pure ReadIdle)
        0
        counter)
    
    arValid = state .==. pure ReadAddr
    arAddr = (\a l -> AxiAR
      { araddr  = a
      , arlen   = l           -- ✅ NOW CORRECT (255 for 256 transfers)
      , arsize  = 6
      , arburst = 1
      , arid    = 0
      }) <$> addr <*> len     -- ✅ Uses latched len!
    
    arHandshake = arValid .&&. arready slaveIn
    rReady = state .==. pure ReadData
    rHandshake = rReady .&&. rvalid slaveIn
    
    masterOut = AxiMasterOut
      { arvalid = arValid
      , ardata  = arAddr
      , rready  = rReady
      , awvalid = pure False
      , awdata  = pure (errorX "not writing")
      , wvalid  = pure False
      , wdataMI   = pure (errorX "not writing")
      , bready  = pure False
      }
    
    dataOut = rdata <$> rdataSI slaveIn
    validOut = rHandshake
    readyOut = state .==. pure ReadIdle
