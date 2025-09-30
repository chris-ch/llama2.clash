module Model.Layers.Attention.AttendSequential
  ( attendHeadSeq ) where

import Clash.Prelude
import Model.Core.Types ( HeadDimension, HeadDimension )
import qualified Model.Layers.Attention.OnlineSoftmax as OnlineSoftmax
    ( softInit, softResult, softStep )
import Model.Numeric.Types (FixedPoint)

dotF :: Vec HeadDimension FixedPoint -> Vec HeadDimension FixedPoint -> FixedPoint
dotF a b = sum (zipWith (*) a b)

attendHeadSeq
  :: HiddenClockResetEnable dom
  => Signal dom Bool                          -- clear/start
  -> Signal dom (Vec HeadDimension FixedPoint)         -- q
  -> Signal dom (Vec HeadDimension FixedPoint)         -- k(t)
  -> Signal dom (Vec HeadDimension FixedPoint)         -- v(t)
  -> Signal dom Bool                          -- lastT
  -> ( Signal dom (Vec HeadDimension FixedPoint)       -- out
     , Signal dom Bool )                      -- done
attendHeadSeq clear qSig kSig vSig lastT =
  (OnlineSoftmax.softResult <$> st, lastT)
 where
  st = mealy
        (\s (cl,q,k,v) ->
           let s' = if cl then OnlineSoftmax.softInit else OnlineSoftmax.softStep s (dotF q k, v)
           in  (s', s'))
        OnlineSoftmax.softInit
        (bundle (clear, qSig, kSig, vSig))
