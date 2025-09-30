module Model.Layers.Attention.OnlineSoftmax
  ( SoftState(..)
  , softInit
  , softStep
  , softResult
  ) where

import Clash.Prelude
import Model.Core.Types (HeadDimension)

data SoftState = SoftState
  { mMax  :: Float
  , denom :: Float
  , numer :: Vec HeadDimension Float
  } deriving (Generic, NFDataX, Show)

softInit :: SoftState
softInit = SoftState { mMax = - (1 / 0), denom = 0, numer = repeat 0 }

softStep :: SoftState -> (Float, Vec HeadDimension Float) -> SoftState
softStep st (x, v) =
  let m  = mMax st
      d0 = denom st
      n0 = numer st
  in if x <= m
       then
         let e  = exp (x - m)
             d1 = d0 + e
             n1 = zipWith (+) n0 (map (* e) v)
         in st { denom = d1, numer = n1 }
       else
         let s  = exp (m - x)
             d1 = d0 * s + 1
             n1 = map (* s) n0
             n2 = zipWith (+) n1 v
         in SoftState { mMax = x, denom = d1, numer = n2 }

softResult :: SoftState -> Vec HeadDimension Float
softResult st = let d = denom st
                in if d == 0 then repeat 0 else map (/ d) (numer st)
