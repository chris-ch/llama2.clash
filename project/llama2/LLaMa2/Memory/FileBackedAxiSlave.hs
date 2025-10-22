module LLaMa2.Memory.FileBackedAxiSlave (
  createFileBackedAxiSlave, ReadState(..)
) where

import Clash.Prelude
import LLaMa2.Memory.AXI
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Vector.Unboxed as V
import Data.Word (Word8)

-- State for read transactions  
data ReadState = RIdle | RBursting
  deriving (Generic, NFDataX, Show, Eq)

-- | File-backed AXI slave
createFileBackedAxiSlave 
  :: forall dom . HiddenClockResetEnable dom
  => BSL.ByteString
  -> AxiMasterOut dom
  -> (AxiSlaveIn dom, Signal dom ReadState)
createFileBackedAxiSlave fileContents masterOut = (slaveIn, readState)
  where
    -- Convert to Vector for fast access
    fileBytes = V.fromList (BSL.unpack fileContents)
    fileLen = V.length fileBytes
    
    -- Helper to convert Word8 to BitVector 8
    word8ToBV :: Word8 -> BitVector 8
    word8ToBV = fromInteger . toInteger
    
    -- Register master outputs to break combinational loop
    arvalid_r = register False (arvalid masterOut)
    ardata_r = register (AxiAR 0 0 0 0 0) (ardata masterOut)
    rready_r = register False (rready masterOut)
    
    -- State machine
    readState = register RIdle nextReadState
    
    -- AR channel: accept when idle
    arReady = (==) <$> readState <*> pure RIdle
    arHandshake = arvalid_r .&&. arReady
    
    -- Latch address and burst length
    latched_r = regEn (0, 0) arHandshake ((\ar -> (araddr ar, fromIntegral (arlen ar) + 1)) <$> ardata_r)
    (latchedAddr, burstLen) = unbundle latched_r
    
    -- Transfer counter
    transferCount = register (0 :: Unsigned 32) nextTransferCount
    
    -- State transition:
    nextReadState = mux (readState .==. pure RIdle)
        (mux arHandshake (pure RBursting) (pure RIdle))  -- No counter
        (mux (isBursting <$> readState)
            (mux (rready_r .&&. (transferCount + 1 .>=. burstLen))
                (pure RIdle)
                (pure RBursting))  -- Just stay in RBursting
            readState)
    
    isBursting RIdle = False
    isBursting RBursting = True
    
    nextTransferCount = mux (readState .==. pure RIdle)
      0
      (mux rready_r
        (transferCount + 1)
        transferCount)
    
    -- R channel: generate data
    rValid = isBursting <$> readState
    
    -- Read file data
    fileData = readFileChunk <$> latchedAddr <*> transferCount
      where
        readFileChunk addr cnt =
          let offset = fromIntegral (addr + (cnt * 64))
              readByte idx = 
                let pos = offset + idx
                in if pos >= fileLen
                   then 0 :: BitVector 8
                   else word8ToBV (fileBytes V.! pos)
          in pack (map readByte (iterateI (+1) (0 :: Int) :: Vec 64 Int))
    
    rLast = ((transferCount + 1) .>=. burstLen) .&&. rValid
    
    rDataOut = (\d l -> AxiR d 0 l 0) <$> fileData <*> rLast
    
    -- Write channels (unused)
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
