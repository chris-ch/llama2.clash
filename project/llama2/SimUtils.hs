module SimUtils
  ( makeCycleCounter
  ) where

import Clash.Prelude

-- | Create a free-running cycle counter. Pass the result to simulation
-- helpers that need a clock reference.
makeCycleCounter :: forall dom. HiddenClockResetEnable dom => Signal dom (Unsigned 32)
makeCycleCounter = cnt
  where cnt = register 0 (cnt + 1)
