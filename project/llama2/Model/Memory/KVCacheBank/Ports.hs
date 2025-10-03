module Model.Memory.KVCacheBank.Ports (mapKVPorts) where

import Clash.Prelude
import Model.Config (BankDepth, HeadDimension)
import Model.Config.KVGroups (KVExpAddress)
import Model.Memory.KVCacheBank (KVBank(..))
import Model.Numeric.Types ( FixedPoint, Activation, Exponent, scalePow2F )
import qualified Model.Memory.Addressing as Addressing

mapKVPorts
  :: ( Signal dom (Index BankDepth)      -- read addr (Stage3)
     , Signal dom Bool                   -- read enable
     , Signal dom (Index BankDepth)      -- write addr (Stage2, mant)
     , Signal dom (Maybe (Index BankDepth, Activation))   -- K mant write
     , Signal dom (Maybe (KVExpAddress, Exponent))        -- K exp write (grouped)
     , Signal dom (Maybe (Index BankDepth, Activation))   -- V mant write
     , Signal dom (Maybe (KVExpAddress, Exponent))        -- V exp write (grouped)
     , KVBank dom)
  -> ( Signal dom FixedPoint              -- K read (dequantized)
     , Signal dom FixedPoint )            -- V read (dequantized)
mapKVPorts (rdAddr, rdEn, wrAddr, wrKm, wrKe, wrVm, wrVe, bank) =
  (kOutF, vOutF)
 where
  letK  = runKeyMantBank   bank
  letKe = runKeyExpBank    bank
  letV  = runValueMantBank bank
  letVe = runValueExpBank  bank

  -- Port A read addresses
  addrMantA = mux rdEn rdAddr (pure 0)
  addrMantB = wrAddr

  -- Compute (t, d) from rdAddr; derive exponent read address
  seqIx = (\a ->
            let hd = natToNum @HeadDimension
            in toEnum (fromEnum a `div` hd))
          <$> addrMantA
  dimIx = Addressing.dimFromBankAddress <$> addrMantA
  addrExpA = Addressing.computeExpAddress <$> seqIx <*> dimIx

  -- Port B write addresses
  addrExpB_K = maybe 0 fst <$> wrKe
  addrExpB_V = maybe 0 fst <$> wrVe

  wrMantA   = pure Nothing
  wrMantB_K = wrKm
  wrMantB_V = wrVm

  wrExpA    = pure Nothing
  wrExpB_K  = wrKe
  wrExpB_V  = wrVe

  (kMantA, _) = letK  (addrMantA, wrMantA) (addrMantB, wrMantB_K)
  (vMantA, _) = letV  (addrMantA, wrMantA) (addrMantB, wrMantB_V)
  (kExpA,  _) = letKe (addrExpA,  wrExpA)  (addrExpB_K, wrExpB_K)
  (vExpA,  _) = letVe (addrExpA,  wrExpA)  (addrExpB_V, wrExpB_V)

  kOutF = (\m e -> fromIntegral m * scalePow2F e 1) <$> kMantA <*> kExpA
  vOutF = (\m e -> fromIntegral m * scalePow2F e 1) <$> vMantA <*> vExpA
