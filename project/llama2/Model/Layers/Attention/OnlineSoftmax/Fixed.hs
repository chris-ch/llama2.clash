module Model.Layers.Attention.OnlineSoftmax.Fixed
  ( SoftStateF(..)
  , softInitF
  , softStepF
  , softResultF
  ) where

import Clash.Prelude
import GHC.Generics (Generic)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.Fixed (expF)
import Model.Core.Types (HeadDimension)

data SoftStateF = SoftStateF
  { mMaxF  :: FixedPoint
  , denomF :: FixedPoint
  , numerF :: Vec HeadDimension FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

softInitF :: SoftStateF
softInitF = SoftStateF { mMaxF = minBound, denomF = 0, numerF = repeat 0 }

softStepF :: SoftStateF -> (FixedPoint, Vec HeadDimension FixedPoint) -> SoftStateF
softStepF st (x, v) =
  let m  = mMaxF st
      d0 = denomF st
      n0 = numerF st
  in if x <= m
       then
         let e  = expF (x - m)
             d1 = d0 + e
             n1 = zipWith (+) n0 (map (* e) v)
         in st { denomF = d1, numerF = n1 }
       else
         let s  = expF (m - x)
             d1 = d0 * s + 1
             n1 = map (* s) n0
             n2 = zipWith (+) n1 v
         in SoftStateF { mMaxF = x, denomF = d1, numerF = n2 }

softResultF :: SoftStateF -> Vec HeadDimension FixedPoint
softResultF st = let d = denomF st
                 in if d == 0 then repeat 0 else map (/ d) (numerF st)
