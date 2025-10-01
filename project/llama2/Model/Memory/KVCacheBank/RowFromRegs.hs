module Model.Memory.KVCacheBank.RowFromRegs
  ( rowsFromRegs ) where

import Clash.Prelude
import Model.Config (SequenceLength, HeadDimension)
import Model.Numeric.Types (FixedPoint)

-- Walk head-dimension to produce a rowValid pulse at the end of each row.
dimCnt
  :: HiddenClockResetEnable dom
  =>  Signal dom Bool -> Signal dom (Index HeadDimension)
dimCnt enS3 = regEn 0 enS3 $
            mux (dimCnt enS3 .==. pure maxBound) (pure 0) (succ <$> dimCnt enS3)

-- Generate (K,V) rows sequentially from register-mirrored memories.
-- This matches the cadence of the BRAM streamer (one element/cycle, rowValid once per row).
rowsFromRegs
  :: HiddenClockResetEnable dom
  => Signal dom Bool                                            -- clearS3 (1-cycle pulse)
  -> Signal dom Bool                                            -- enS3
  -> Signal dom (Index SequenceLength)                          -- pos
  -> Signal dom (Vec SequenceLength (Vec HeadDimension FixedPoint)) -- K all rows
  -> Signal dom (Vec SequenceLength (Vec HeadDimension FixedPoint)) -- V all rows
  -> ( Signal dom (Vec HeadDimension FixedPoint)                -- K row
     , Signal dom (Vec HeadDimension FixedPoint)                -- V row
     , Signal dom Bool                                          -- rowValid
     , Signal dom Bool )                                        -- lastT
rowsFromRegs clearS3 enS3 posSig kAllSig vAllSig =
  (kRow, vRow, rowValid, lastT)
 where

  rowEnd = enS3 .&&. (dimCnt enS3 .==. pure maxBound)

  -- Row index t; advance at row boundaries up to pos
  tCnt = mealy
           (\t (cl,enEnd,pos) ->
              let t'
                    | cl = 0
                    | enEnd && t < pos = succ t
                    | otherwise = t
              in (t', t'))
           0
           (bundle (clearS3, rowEnd, posSig))

  -- Present full rows continuously (attendHeadSeq will step only on rowValid)
  kRow = (!!) <$> kAllSig <*> tCnt
  vRow = (!!) <$> vAllSig <*> tCnt

  rowValid = rowEnd
  lastT    = rowValid .&&. (tCnt .==. posSig)
