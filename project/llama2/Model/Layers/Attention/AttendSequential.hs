-- ===== project/llama2/Model/Layers/Attention/AttendSequential.hs =====
module Model.Layers.Attention.AttendSequential
  ( attendHeadSeq ) where

import Clash.Prelude
import Model.Config (HeadDimension)
import qualified Model.Layers.Attention.OnlineSoftmax as OnlineSoftmax
  ( softInit, softResult, softStep )
import Model.Numeric.Types (FixedPoint)

-- Dot product in FixedPoint
dotF :: Vec HeadDimension FixedPoint -> Vec HeadDimension FixedPoint -> FixedPoint
dotF a b = sum (zipWith (*) a b)

-- Sequential attention for one head:
-- - clear: one-cycle pulse when entering Stage3 (resets softmax state)
-- - stepEn: high exactly when a full (K,V) row is ready (rowValid)
-- - q, kRow, vRow sampled when stepEn is high
-- - lastT: asserted together with stepEn for the last row (t == pos)
--
-- Output:
-- - out: final attended vector (stable except when lastT was just asserted)
-- - done: one-cycle pulse aligned with lastT
attendHeadSeq
  :: HiddenClockResetEnable dom
  => Signal dom Bool                          -- clear
  -> Signal dom Bool                          -- stepEn (rowValid)
  -> Signal dom (Vec HeadDimension FixedPoint)         -- q
  -> Signal dom (Vec HeadDimension FixedPoint)         -- kRow
  -> Signal dom (Vec HeadDimension FixedPoint)         -- vRow
  -> Signal dom Bool                          -- lastT (valid only when stepEn)
  -> ( Signal dom (Vec HeadDimension FixedPoint)       -- out
     , Signal dom Bool )                      -- done
attendHeadSeq clear stepEn qSig kSig vSig lastT =
  (OnlineSoftmax.softResult <$> st, done)
 where
  -- Same scale as combinational attention
  scale :: FixedPoint
  scale = realToFrac (1.0 / sqrt (fromIntegral (natToNum @HeadDimension) :: Double))

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

  done = stepEn .&&. lastT
