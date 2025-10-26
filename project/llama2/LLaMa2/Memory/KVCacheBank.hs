module LLaMa2.Memory.KVCacheBank (
  writePulseGenerator
) where

import Clash.Prelude

-- one-pulse generator (rising edge of 'en')
writePulseGenerator
  :: HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ en (Level during Stage2)
  -> ( Signal dom Bool  -- ^ wrPulse (1 cycle on Stage2 entry)
     , Signal dom Bool) -- ^ donePulse (1 cycle, same as wrPulse by default)
writePulseGenerator enSig =
  let enPrev   = register False enSig
      pulse    = enSig .&&. not <$> enPrev
  in (pulse, pulse)
