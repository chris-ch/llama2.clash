{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Redundant bracket" #-}
module LLaMa2.Memory.SimAxiSlave (
  createSimAxiSlave
) where

import Clash.Prelude
import LLaMa2.Memory.AXI

-- State machine for handling read transactions
data ReadState = RIdle | RBurst (Unsigned 8) (Unsigned 32)
  deriving (Generic, NFDataX, Show, Eq)
    
-- | Simulated AXI slave that responds to read requests with test patterns
-- Useful for testing weight streaming without actual hardware
createSimAxiSlave :: forall dom . HiddenClockResetEnable dom
  => AxiMasterOut dom
  -> AxiSlaveIn dom
createSimAxiSlave masterOut = slaveIn
  where
    readState = register RIdle nextReadState
    
    -- Accept read address
    arReady = readState .==. pure RIdle
    arHandshake = arvalid masterOut .&&. arReady
    
    -- Latch address and burst length when accepted
    latched = regEn (0, 0) arHandshake 
      ((\ar -> (araddr ar, arlen ar)) <$> ardata masterOut)
    (latchedAddr, latchedLen) = unbundle latched
    
    -- Current transfer in burst
    transferCount = register (0 :: Unsigned 8) nextTransferCount
    
    -- Generate test data pattern based on address
    -- Pattern: fill with address bits repeated
    testData :: Signal dom (BitVector 512)
    testData = (\addr cnt -> 
      let basePattern = resize (pack addr) :: BitVector 8
          byte = basePattern + resize (pack cnt)
          byteVec = unpack byte :: Vec 8 Bit
          repeated = replicate d64 byteVec :: Vec 64 (Vec 8 Bit)
          flattened = concat repeated :: Vec 512 Bit
      in pack flattened
      ) <$> latchedAddr <*> transferCount

    -- State transitions
    nextReadState = case_readState <$> readState <*> arHandshake 
                      <*> (rready masterOut) <*> transferCount <*> latchedLen
    
    case_readState RIdle True _ _ _ = RBurst 0 0  -- Start burst
    case_readState (RBurst cnt _) _ True tcnt len 
      | tcnt == len = RIdle  -- Burst complete
      | otherwise   = RBurst (cnt + 1) 0  -- Next transfer
    case_readState st _ _ _ _ = st  -- Stay in current state
    
    nextTransferCount = mux (readState .==. pure RIdle)
      0
      (mux (rready masterOut)
        (transferCount + 1)
        transferCount)
    
    -- R channel outputs
    rValid = case_rvalid <$> readState
    case_rvalid RIdle = False
    case_rvalid (RBurst _ _) = True
    
    rLast = (transferCount .==. latchedLen) .&&. rValid
    
    rDataOut = (\d l -> AxiR 
      { rdata = d
      , rresp = 0  -- OKAY
      , rlast = l
      , rid = 0
      }) <$> testData <*> rLast
    
    -- Write channels (not implemented - always ready, never valid)
    awReady = pure True
    wReady = pure True
    bValid = pure False
    bData = pure (AxiB 0 0)
    
    slaveIn = AxiSlaveIn
      { arready = arReady
      , rvalid = rValid
      , rdataSI = rDataOut
      , awready = awReady
      , wready = wReady
      , bvalid = bValid
      , bdata = bData
      }
