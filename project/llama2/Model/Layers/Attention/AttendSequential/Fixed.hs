module Model.Layers.Attention.AttendSequential.Fixed
  ( attendHeadSeqF ) where

import Clash.Prelude
import Model.Core.Types (HeadDimension)
import Model.Numeric.Types (F)
import Model.Layers.Attention.OnlineSoftmax.Fixed

dotF :: Vec HeadDimension F -> Vec HeadDimension F -> F
dotF a b = sum (zipWith (*) a b)

attendHeadSeqF
  :: HiddenClockResetEnable dom
  => Signal dom Bool                          -- clear/start
  -> Signal dom (Vec HeadDimension F)         -- q
  -> Signal dom (Vec HeadDimension F)         -- k(t)
  -> Signal dom (Vec HeadDimension F)         -- v(t)
  -> Signal dom Bool                          -- lastT
  -> ( Signal dom (Vec HeadDimension F)       -- out
     , Signal dom Bool )                      -- done
attendHeadSeqF clear qSig kSig vSig lastT =
  (softResultF <$> st, lastT)
 where
  st = mealy
        (\s (cl,q,k,v) ->
           let s' = if cl then softInitF else softStepF s (dotF q k, v)
           in  (s', s'))
        softInitF
        (bundle (clear, qSig, kSig, vSig))
