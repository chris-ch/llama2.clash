module Model.Memory.KVCacheBank (
  writeSequencer
) where

import Clash.Prelude

import Model.Config
  ( HeadDimension
   )

-- Write sequencer: emits mant per cycle; exponent at each group start
writeSequencer :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
writeSequencer enSig = doneSig
 where
  -- Head-dim counter
  dimCnt :: Signal dom (Index HeadDimension)
  dimCnt     = register 0 nextDimCnt
  nextDimCnt :: Signal dom (Index HeadDimension)
  nextDimCnt = mux enSig (fmap (\d -> if d == maxBound then 0 else succ d) dimCnt) (pure 0)
  atLastDim  = (== maxBound) <$> dimCnt

  doneSig :: Signal dom Bool
  doneSig    = (&&) <$> enSig <*> atLastDim
