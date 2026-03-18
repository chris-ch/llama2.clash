module LLaMa2.Memory.ActivationBRAM
  ( ActivationBramReadPort (..)
  , ActivationBramWritePort (..)
  , activationBram
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.LayerData (ActivationBramAddr, ActivationBramDepth)
import LLaMa2.Memory.DualPortRAM (trueDualPortRam)

-- | Read port: present an address each cycle; data arrives the next cycle.
data ActivationBramReadPort dom = ActivationBramReadPort
  { rdAddr :: Signal dom ActivationBramAddr }

-- | Write port: Nothing means no write this cycle.
data ActivationBramWritePort dom = ActivationBramWritePort
  { wrAddr :: Signal dom ActivationBramAddr
  , wrData :: Signal dom (Maybe FixedPoint) }

-- | Dual-port activation BRAM.
--
-- Port A is the read port; port B is the write port.
-- Read latency is 1 cycle (synchronous read).
-- Simultaneous reads and writes to different slot addresses are safe.
activationBram
  :: HiddenClockResetEnable dom
  => ActivationBramReadPort dom
  -> ActivationBramWritePort dom
  -> Signal dom FixedPoint   -- ^ read data, 1-cycle latency
activationBram (ActivationBramReadPort ra) (ActivationBramWritePort wa wd) =
  fst $ trueDualPortRam
    ra
    (pure Nothing)           -- port A: read-only
    wa
    (toWrite <$> wa <*> wd)
  where
    toWrite :: ActivationBramAddr -> Maybe FixedPoint -> Maybe (ActivationBramAddr, FixedPoint)
    toWrite addr (Just d) = Just (addr, d)
    toWrite _    Nothing  = Nothing
