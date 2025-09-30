module Model.Helpers.EnableWrappers (holdWhen) where

import Clash.Prelude

-- Hold last output value when en = False
holdWhen :: (HiddenClockResetEnable dom, NFDataX a)
         => Signal dom Bool
         -> Signal dom a
         -> Signal dom a
holdWhen = regEn (deepErrorX "holdWhen init")
