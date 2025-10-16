module LLaMa2.Layers.Attention.AttentionHead
  ( attendHead ) where

import Clash.Prelude
import LLaMa2.Config (HeadDimension)
import LLaMa2.Numeric.Types (FixedPoint)

import qualified LLaMa2.Layers.Attention.OnlineSoftmax as OnlineSoftmax
  ( softInit, softResult, softStep )

dotF :: Vec HeadDimension FixedPoint -> Vec HeadDimension FixedPoint -> FixedPoint
dotF a b = sum (zipWith (*) a b)

attendHead :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec HeadDimension FixedPoint)
  -> Signal dom (Vec HeadDimension FixedPoint)
  -> Signal dom (Vec HeadDimension FixedPoint)
  -> Signal dom Bool
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool )
attendHead clear stepEn qSig kSig vSig lastT =
  (OnlineSoftmax.softResult <$> st, stepEn .&&. lastT)
 where
  scale :: FixedPoint
  scale = realToFrac (1.0 / sqrt ((natToNum @HeadDimension) :: Double))

  stepInput =
    mux stepEn
      (Just <$> bundle ( (* scale) <$> (dotF <$> qSig <*> kSig), vSig))
      (pure Nothing)

  st = mealy
        (\s (cl, inpM) ->
           let s0 = if cl then OnlineSoftmax.softInit else s
           in case inpM of
                Nothing   -> (s0, s0)
                Just pair -> let s1 = OnlineSoftmax.softStep s0 pair
                             in  (s1, s1))
        OnlineSoftmax.softInit
        (bundle (clear, stepInput))
