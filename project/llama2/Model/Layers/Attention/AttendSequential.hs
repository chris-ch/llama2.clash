module Model.Layers.Attention.AttendSequential
  ( attendHeadSeq ) where

import Clash.Prelude
import Model.Core.Types (HeadDimension)
import Model.Layers.Attention.OnlineSoftmax

dot :: Vec HeadDimension Float -> Vec HeadDimension Float -> Float
dot a b = sum (zipWith (*) a b)

attendHeadSeq
  :: HiddenClockResetEnable dom
  => Signal dom Bool                          -- clear/start (1-cycle at Stage3 entry)
  -> Signal dom (Vec HeadDimension Float)     -- q (constant during Stage3)
  -> Signal dom (Vec HeadDimension Float)     -- k(t)
  -> Signal dom (Vec HeadDimension Float)     -- v(t)
  -> Signal dom Bool                          -- lastT (1 when t == pos)
  -> ( Signal dom (Vec HeadDimension Float)   -- out (valid when done)
     , Signal dom Bool )                      -- done pulse
attendHeadSeq clear qSig kSig vSig lastT =
  (softResult <$> st, lastT)
 where
  st = mealy
        (\s (cl,q,k,v) ->
           let s' = if cl then softInit else softStep s (dot q k, v)
           in  (s', s'))
        softInit
        (bundle (clear, qSig, kSig, vSig))