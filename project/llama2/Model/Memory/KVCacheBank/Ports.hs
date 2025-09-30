module Model.Memory.KVCacheBank.Ports (mapKVPorts) where

import Clash.Prelude
import Model.Core.Types (BankDepth, SequenceLength, HeadDimension)
import Model.Memory.KVCacheBank (KvBank(..))
import Model.Numeric.Types ( F, Act, ExpS, scalePow2F )

mapKVPorts
  :: HiddenClockResetEnable dom
  => ( Signal dom (Index BankDepth)      -- read addr (Stage3)
     , Signal dom Bool                   -- read enable
     , Signal dom (Index BankDepth)      -- write addr (Stage2)  -- unused here
     , Signal dom (Maybe (Index BankDepth, Act))   -- K mant write
     , Signal dom (Maybe (Index SequenceLength, ExpS)) -- K exp write
     , Signal dom (Maybe (Index BankDepth, Act))   -- V mant write
     , Signal dom (Maybe (Index SequenceLength, ExpS)) -- V exp write
     , KvBank dom)
  -> ( Signal dom F                      -- K read (dequantized)
     , Signal dom F )                    -- V read (dequantized)
mapKVPorts (rdAddr, rdEn, wrAddr, wrKm, wrKe, wrVm, wrVe, bank) =
  (kOutF, vOutF)
 where
  -- Split bank
  letK  = runKeyMantBank bank
  letKe = runKeyExpBank  bank
  letV  = runValueMantBank bank
  letVe = runValueExpBank bank

  -- Addresses
  addrMantA = mux rdEn rdAddr (pure 0)
  addrMantB = wrAddr

  -- Sequence index = rdAddr // HeadDimension
  seqIx = (\a ->
            let hd = natToNum @HeadDimension :: Int
            in toEnum (fromEnum a `div` hd))
          <$> addrMantA

  addrExpA = mux rdEn seqIx (pure 0)
  addrExpB = maybe 0 fst <$> wrKe  -- write side provides (pos, exp)

  wrMantA = pure Nothing
  wrMantB_K = wrKm
  wrMantB_V = wrVm

  wrExpA = pure Nothing
  wrExpB_K = wrKe
  wrExpB_V = wrVe

  (kMantA, _) = letK  (addrMantA, wrMantA) (addrMantB, wrMantB_K)
  (vMantA, _) = letV  (addrMantA, wrMantA) (addrMantB, wrMantB_V)
  (kExpA,  _) = letKe (addrExpA,  wrExpA)  (addrExpB,  wrExpB_K)
  (vExpA,  _) = letVe (addrExpA,  wrExpA)  (addrExpB,  wrExpB_V)

  -- Dequantize element-wise: out = mant * 2^exp
  kOutF = (\m e -> fromIntegral m * scalePow2F e 1) <$> kMantA <*> kExpA
  vOutF = (\m e -> fromIntegral m * scalePow2F e 1) <$> vMantA <*> vExpA
