module Model.Memory.KVCacheBank.RowStreamer
  ( kvRowStreamer
  ) where

import Clash.Prelude
import Model.Config ( SequenceLength, HeadDimension, BankDepth )
import Model.Memory.KVCacheBank (KvBank(..))
import Model.Memory.KVCacheBank.Ports (mapKVPorts)
import qualified Model.Memory.Addressing as Addressing
import Model.Numeric.Types (FixedPoint, Act, ExpS)

-- Stream dequantized K/V rows from BRAM during Stage3:
-- - Issues element addresses for (t, d) with d = 0..HeadDim-1
-- - Assembles full Vec HeadDimension for K and V
-- - rowValid is high for one cycle when the row is complete
-- - lastT is high together with rowValid when t == pos (inclusive)
--
-- Ports:
--   clearS3  : 1-cycle pulse when entering Stage3
--   enS3     : Stage3 active (read enable)
--   posSig   : current (inclusive) sequence position for this attention
--   wr*      : pass-through of Stage2 write channels, already gated by caller
kvRowStreamer
  :: HiddenClockResetEnable dom
  => KvBank dom
  -> Signal dom Bool                                   -- clearS3
  -> Signal dom Bool                                   -- enS3
  -> Signal dom (Index SequenceLength)                 -- pos
  -> Signal dom (Index BankDepth)                      -- wrAddr (for mant ports)
  -> Signal dom (Maybe (Index BankDepth, Act))         -- K mant write (gated by caller)
  -> Signal dom (Maybe (Index SequenceLength, ExpS))   -- K exp write (gated by caller)
  -> Signal dom (Maybe (Index BankDepth, Act))         -- V mant write (gated by caller)
  -> Signal dom (Maybe (Index SequenceLength, ExpS))   -- V exp write (gated by caller)
  -> ( Signal dom (Vec HeadDimension FixedPoint)       -- K row
     , Signal dom (Vec HeadDimension FixedPoint)       -- V row
     , Signal dom Bool                                 -- rowValid
     , Signal dom Bool )                               -- lastT
kvRowStreamer bank clearS3 enS3 posSig wrAddr kMantWr kExpWr vMantWr vExpWr =
  (kRowOut, vRowOut, rowValid, lastT)
 where
  -- Walk (t, d)
  dimCnt = regEn 0 enS3 $
             mux (dimCnt .==. pure maxBound) (pure 0) (succ <$> dimCnt)

  -- Advance t at the end of each assembled row, but do not run past pos
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

  -- Map to element address stream
  rdAddr = Addressing.computeBankAddress <$> tCnt <*> dimCnt

  -- Wire up BRAM ports and dequantize elements with a 1-cycle read latency
  (kElF, vElF) = mapKVPorts
                   ( rdAddr, enS3
                   , wrAddr, kMantWr, kExpWr, vMantWr, vExpWr
                   , bank)

  -- Align the write index with the 1-cycle BRAM read latency
  dimCntD = register 0 dimCnt
  tCntD   = register 0 tCnt

  -- Assemble full rows by writing the returned element into the right slot
  kRowOut = mealy
              (\row (cl,en,i,x) ->
                 let row0 = if cl then repeat 0 else row
                     row1 = if en  then replace i x row0 else row0
                 in (row1, row1))
              (repeat 0)
              (bundle (clearS3, enS3, dimCntD, kElF))

  vRowOut = mealy
              (\row (cl,en,i,x) ->
                 let row0 = if cl then repeat 0 else row
                     row1 = if en  then replace i x row0 else row0
                 in (row1, row1))
              (repeat 0)
              (bundle (clearS3, enS3, dimCntD, vElF))

  rowValid = enS3 .&&. (dimCntD .==. pure maxBound)
  lastT    = rowValid .&&. (tCntD .==. posSig)
