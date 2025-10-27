{-# LANGUAGE DerivingVia #-}
module Simulation.FileBackedAxiSlave
  ( createFileBackedAxiSlave
  ) where

import Clash.Prelude
import qualified Data.ByteString.Lazy as BSL
import Data.ByteString.Lazy (ByteString)

import LLaMa2.Memory.AXI.Slave (AxiSlaveIn (..))
import LLaMa2.Memory.AXI.Master (AxiMasterOut (..))
import LLaMa2.Memory.AXI.Types (AxiAR (..), AxiB (..), AxiR (..))

data StateRAM = SIdle | SBurst
  deriving (Generic, NFDataX, Eq, Show)

newtype Address = Address (Unsigned 32)
  deriving stock (Eq, Show)
  deriving newtype (NFDataX, BitPack, Default, Num, Ord)

newtype BeatsLeft = BeatsLeft (Unsigned 8)
  deriving stock (Eq, Show)
  deriving newtype (NFDataX, BitPack, Default, Num, Ord)

type StateFSM = (StateRAM, Address, BeatsLeft)

--------------------------------------------------------------------------------
-- Helpers
--------------------------------------------------------------------------------

-- | Read a 512-bit (64-byte) word from a ByteString at a given byte address.
getWord512FromFile :: ByteString -> Address -> BitVector 512
getWord512FromFile bs addr =
  let base = addr
      fileLen = fromIntegral (BSL.length bs) :: Address
      go :: BitVector 512 -> Index 64 -> BitVector 512
      go acc i =
        let byteIdxInt = base + fromIntegral i
            Address byteIdx64 = byteIdxInt
            byteVal :: BitVector 8
            byteVal | byteIdxInt < fileLen = fromIntegral (BSL.index bs (fromIntegral byteIdx64))
                    | otherwise = 0
        in (acc `shiftL` 8) .|. resize byteVal
  in foldl go 0 indicesI

--------------------------------------------------------------------------------
-- File-backed AXI slave (read-only)
--------------------------------------------------------------------------------

createFileBackedAxiSlave ::
  forall dom.
  (HiddenClockResetEnable dom) =>
  ByteString ->
  AxiMasterOut dom ->
  AxiSlaveIn dom
createFileBackedAxiSlave modelBin AxiMasterOut{..} = slaveOut
 where
  -- Handshake: we are always ready
  arHandshake = arvalid .&&. pure True

  initState :: StateFSM
  initState = (SIdle, Address 0, BeatsLeft 0)

  -- FSM: respond to address handshake, then stream data beats
  step :: StateFSM
       -> (AxiAR, Bool)               -- (AR channel, handshake)
       -> (StateFSM, (StateRAM, Address, Bool, Bool))
  step (SIdle, _, _) (ar, handshake)
    | handshake =
        let len  = arlen ar + 1
            addr = Address $ araddr ar
        in  ((SBurst, addr, BeatsLeft $ len - 1), (SIdle, addr, False, False))
    | otherwise = ((SIdle, Address 0, 0), (SIdle, Address 0, False, False))
  step (SBurst, addr, beatsLeft) (_, _) =
    let nextAddr = addr + Address 64          -- 512-bit word = 64 bytes
        nextBeats = beatsLeft - 1
        done = beatsLeft == 0
        nextState = if done then SIdle else SBurst
    in  ((nextState, nextAddr, nextBeats), (SBurst, addr, True, done))

  -- Run FSM
  (stateRAM, rAddrRaw, rvalidNow, rlastNow) = mealyB step initState (ardata, arHandshake)

  -- One-cycle latency to match BRAM timing
  rAddr  = register 0 rAddrRaw
  rvalid = register False rvalidNow
  rlast  = register False rlastNow

  -- Data fetch (combinational)
  rdataWord = getWord512FromFile modelBin <$> rAddr

  -- Tie-offs for unused write channels
  awready = pure False
  wready  = pure False
  bvalid  = pure False
  bdata   = pure (AxiB 0 0)

  -- Outputs
  slaveOut =
    AxiSlaveIn
      { arready = pure True
      , rvalid  = rvalid
      , rdata = AxiR <$> rdataWord <*> pure 0 <*> rlast <*> pure 0
      , awready = awready
      , wready  = wready
      , bvalid  = bvalid
      , bdata   = bdata
      }
