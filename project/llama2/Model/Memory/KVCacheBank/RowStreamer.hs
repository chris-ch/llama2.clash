module Model.Memory.KVCacheBank.RowStreamer
  ( kvRowStreamer
  ) where

import Clash.Prelude
import Model.Config (SequenceLength, HeadDimension, BankDepth)
import Model.Config.KVGroups (KVExpAddress)
import Model.Memory.KVCacheBank (KVBank(..))
import Model.Memory.KVCacheBank.Ports (mapKVPorts)
import qualified Model.Memory.Addressing as Addressing
import Model.Numeric.Types (FixedPoint, Activation, Exponent)

kvRowStreamer
  :: HiddenClockResetEnable dom
  => KVBank dom
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Index BankDepth)
  -> Signal dom (Maybe (Index BankDepth, Activation))
  -> Signal dom (Maybe (KVExpAddress, Exponent))
  -> Signal dom (Maybe (Index BankDepth, Activation))
  -> Signal dom (Maybe (KVExpAddress, Exponent))
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool )
kvRowStreamer bank clearS3 enS3 posSig wrAddr kMantWr kExpWr vMantWr vExpWr =
  (kRowOut, vRowOut, rowValid, lastT)
 where
  -- Head-dim counter
  dimCnt =
    mealy
      (\d (cl,en) ->
         let d'
               | cl = 0
               | en = (if d == maxBound then 0 else succ d)
               | otherwise = d
         in (d', d'))
      (0 :: Index HeadDimension)
      (bundle (clearS3, enS3))

  advanceRow = enS3 .&&. (dimCnt .==. pure maxBound)

  tCnt = mealy
           (\t (cl,enEnd,pos) ->
              let t'
                    | cl = 0
                    | enEnd && t < pos = succ t
                    | otherwise = t
              in (t', t'))
           0
           (bundle (clearS3, advanceRow, posSig))

  rdAddr = Addressing.computeBankAddress <$> tCnt <*> dimCnt

  (kElF, vElF) = mapKVPorts
                   ( rdAddr, enS3
                   , wrAddr, kMantWr, kExpWr, vMantWr, vExpWr
                   , bank)

  dimCntD = register 0 dimCnt
  tCntD   = register 0 tCnt

  writeEn = enS3 .&&. (not <$> clearS3)

  kRowOut = mealy
              (\row (cl,we,i,x) ->
                 let row0 = if cl then repeat 0 else row
                     row1 = if we then replace i x row0 else row0
                 in (row1, row1))
              (repeat 0)
              (bundle (clearS3, writeEn, dimCntD, kElF))

  vRowOut = mealy
              (\row (cl,we,i,x) ->
                 let row0 = if cl then repeat 0 else row
                     row1 = if we then replace i x row0 else row0
                 in (row1, row1))
              (repeat 0)
              (bundle (clearS3, writeEn, dimCntD, vElF))

  rowValid = writeEn .&&. (dimCntD .==. pure maxBound)
  lastT    = rowValid .&&. (tCntD .==. posSig)
