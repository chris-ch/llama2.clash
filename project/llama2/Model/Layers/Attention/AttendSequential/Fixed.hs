module Model.Layers.Attention.AttendSequential.Fixed
  ( attendHeadSeqF ) where

import Clash.Prelude
import Model.Core.Types (HeadDimension)
import Model.Numeric.Types (FixedPoint)
import Model.Layers.Attention.OnlineSoftmax.Fixed

dotF :: Vec HeadDimension FixedPoint -> Vec HeadDimension FixedPoint -> FixedPoint
dotF a b = sum (zipWith (*) a b)

attendHeadSeqF
  :: HiddenClockResetEnable dom
  => Signal dom Bool                          -- clear/start
  -> Signal dom (Vec HeadDimension FixedPoint)         -- q
  -> Signal dom (Vec HeadDimension FixedPoint)         -- k(t)
  -> Signal dom (Vec HeadDimension FixedPoint)         -- v(t)
  -> Signal dom Bool                          -- lastT
  -> ( Signal dom (Vec HeadDimension FixedPoint)       -- out
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
