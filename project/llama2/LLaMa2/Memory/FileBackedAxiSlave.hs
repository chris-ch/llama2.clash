module LLaMa2.Memory.FileBackedAxiSlave (
  createFileBackedAxiSlave
) where

import Clash.Prelude
import LLaMa2.Memory.AXI
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Vector.Unboxed as V
import Data.Word (Word8)

-- State machine for handling read transactions
data ReadState = RIdle | RBurst (Unsigned 32)
  deriving (Generic, NFDataX, Show, Eq)

-- | File-backed AXI slave that serves real model weights
-- Reads from the actual model binary file
createFileBackedAxiSlave 
  :: forall dom . HiddenClockResetEnable dom
  => BSL.ByteString              -- Model file contents
  -> AxiMasterOut dom
  -> (AxiSlaveIn dom, Signal dom (Unsigned 32), Signal dom Bool)
createFileBackedAxiSlave fileContents masterOut = (slaveIn, arHandshakeCount, arReady)
  where
    -- Convert file to Vector for O(1) indexed access
    fileBytes = V.fromList (BSL.unpack fileContents)
    
    -- Helper to convert Word8 to BitVector 8
    word8ToBV :: Word8 -> BitVector 8
    word8ToBV w = fromInteger (toInteger w)
    
    readState = register RIdle nextReadState
    
    -- Accept read address when idle
    arReady = case_arready <$> readState
      where
        case_arready RIdle = True
        case_arready _ = False
    
    arHandshake = arvalid masterOut .&&. arReady
    arHandshakeCount = register (0 :: Unsigned 32) nextCount
      where
      nextCount = mux arHandshake (arHandshakeCount + 1) arHandshakeCount
    -- Latch address and burst length when accepted
    latched = regEn (0, 0) arHandshake 
      ((\ar -> (araddr ar, fromIntegral (arlen ar) + 1)) <$> ardata masterOut)
    (latchedAddr, burstLen) = unbundle latched
    
    -- Current transfer in burst
    transferCount = register (0 :: Unsigned 32) nextTransferCount
    
    -- Read actual file data based on address and transfer count
    fileData :: Signal dom (BitVector 512)
    fileData = readFileChunk <$> latchedAddr <*> transferCount
      where
        readFileChunk addr cnt = 
          let offset = addr + (cnt * 64)  -- 64 bytes per transfer
              -- Read 64 bytes from file starting at offset
              bytes = map (readByte offset) (indicesI :: Vec 64 (Index 64))
          in pack bytes
        
        -- Read a single byte from file, return 0 if out of bounds
        readByte offset idx =
          let pos = fromIntegral (offset + fromIntegral idx)
          in if pos >= V.length fileBytes
             then 0 :: BitVector 8
             else word8ToBV (fileBytes V.! pos)

    -- State transitions (same as SimAxiSlave)
    nextReadState = case_nextState <$> readState <*> arHandshake 
                      <*> rReady <*> transferCount <*> burstLen
      where
        case_nextState RIdle True _ _ _ = RBurst 0
        case_nextState (RBurst _) _ True cnt len 
          | cnt + 1 >= len = RIdle
          | otherwise      = RBurst (cnt + 1)
        case_nextState st _ _ _ _ = st
    
    nextTransferCount = case_nextCnt <$> readState <*> rReady <*> transferCount
      where
        case_nextCnt RIdle _ _ = 0
        case_nextCnt (RBurst _) True cnt = cnt + 1
        case_nextCnt _ _ cnt = cnt
    
    -- R channel
    rValid = case_rvalid <$> readState
      where
        case_rvalid RIdle = False
        case_rvalid (RBurst _) = True
    
    rReady = rready masterOut
    
    rLast = checkLast <$> readState <*> transferCount <*> burstLen
      where
        checkLast (RBurst _) cnt len = cnt + 1 >= len
        checkLast _ _ _ = False
    
    rDataOut = mkAxiR <$> fileData <*> rLast
      where
        mkAxiR d l = AxiR 
          { rdata = d
          , rresp = 0
          , rlast = l
          , rid = 0
          }
    
    -- Write channels (not used)
    awReady = pure True
    wReady = pure True
    bValid = register False (wvalid masterOut .&&. wReady)
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
