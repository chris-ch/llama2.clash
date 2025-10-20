module LLaMa2.Memory.KVCacheBank (
  writeSequencer,
  writePulseGenerator
) where

import Clash.Prelude
import LLaMa2.Config (HeadDimension)

-- Existing (unchanged) counter-based sequencer (if you still want it):
writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
writeSequencer enSig = doneSig
 where
  dimCnt     = register (0 :: Index HeadDimension) nextDimCnt
  nextDimCnt = mux enSig (succ <$> dimCnt) (pure 0)
  atLastDim  = (== maxBound) <$> dimCnt
  doneSig    = (&&) <$> enSig <*> atLastDim

-- New: one-pulse generator (rising edge of 'en')
writePulseGenerator
  :: HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ en (Level during Stage2)
  -> ( Signal dom Bool  -- ^ wrPulse (1 cycle on Stage2 entry)
     , Signal dom Bool) -- ^ donePulse (1 cycle, same as wrPulse by default)
writePulseGenerator enSig =
  let enPrev   = register False enSig
      pulse    = enSig .&&. not <$> enPrev
  in (pulse, pulse)
