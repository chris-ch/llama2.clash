module Model.Helpers.EnableWrappers (holdWhen) where

import Clash.Prelude

-- Hold last output value when en = False
-- Requires an explicit reset value.
holdWhen
  :: (HiddenClockResetEnable dom, NFDataX a)
  => a                  -- ^ reset/initial value
  -> Signal dom Bool    -- ^ en
  -> Signal dom a       -- ^ in
  -> Signal dom a
holdWhen = regEn
