module LLaMa2.Layer.Attention.OnlineSoftmax
  ( softInit, softStep, softResult, SoftState(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Fixed (expF)
import LLaMa2.Config (HeadDimension)

data SoftState = SoftStateF
  { mMaxF  :: FixedPoint
  , denomF :: FixedPoint
  , numerF :: Vec HeadDimension FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

softInit :: SoftState
softInit = SoftStateF { mMaxF = minBound, denomF = 0, numerF = repeat 0 }

softStep :: SoftState -> (FixedPoint, Vec HeadDimension FixedPoint) -> SoftState
softStep st (x, v) =
  let m  = mMaxF st
      d0' = denomF st
      n0 = numerF st
  in if x <= m
       then
         let e  = expF (x - m)
             d1' = d0' + e
             n1 = zipWith (+) n0 (map (* e) v)
         in st { denomF = d1', numerF = n1 }
       else
         let s  = expF (m - x)
             d1' = d0' * s + 1
             n1 = map (* s) n0
             n2 = zipWith (+) n1 v
         in SoftStateF { mMaxF = x, denomF = d1', numerF = n2 }

softResult :: SoftState -> Vec HeadDimension FixedPoint
softResult st = let d = denomF st
                 in if d == 0 then repeat 0 else map (/ d) (numerF st)
