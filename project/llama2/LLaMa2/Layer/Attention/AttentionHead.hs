module LLaMa2.Layer.Attention.AttentionHead
  ( attentionHead ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig  (HeadDimension)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (expF)

data SoftState = SoftStateF
  { mMaxF  :: FixedPoint
  , denomF :: FixedPoint
  , numerF :: Vec HeadDimension FixedPoint
  } deriving (Generic, NFDataX, Show, Eq)

dotProduct :: Vec HeadDimension FixedPoint -> Vec HeadDimension FixedPoint -> FixedPoint
dotProduct a b = sum (zipWith (*) a b)

attentionHead :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec HeadDimension FixedPoint)
  -> Signal dom (Vec HeadDimension FixedPoint)
  -> Signal dom (Vec HeadDimension FixedPoint)
  -> Signal dom Bool
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool )
attentionHead clear stepEn qSig kSig vSig lastT =
  (softResult <$> st, stepEn .&&. lastT)
 where
  scale :: FixedPoint
  scale = realToFrac (1.0 / sqrt ((natToNum @HeadDimension) :: Double))

  stepInput =
    mux stepEn
      (Just <$> bundle ( (* scale) <$> (dotProduct <$> qSig <*> kSig), vSig))
      (pure Nothing)

  st = mealy
        (\s (cl, inpM) ->
           let s0 = if cl then softInit else s
           in case inpM of
                Nothing   -> (s0, s0)
                Just pair -> let s1 = softStep s0 pair
                             in  (s1, s1))
        softInit
        (bundle (clear, stepInput))

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
