module Model.Memory.KVCacheBank.Ports (mapKVPorts) where

import Clash.Prelude
import Model.Core.Types (BankDepth)
import Model.Memory.KVCacheBank (KvBank(..))

mapKVPorts
  :: HiddenClockResetEnable dom
  => ( Signal dom (Index BankDepth)                -- read addr (Stage3)
     , Signal dom Bool                             -- read enable
     , Signal dom (Index BankDepth)                -- write addr (Stage2)
     , Signal dom (Maybe (Index BankDepth, Float)) -- K write
     , Signal dom (Maybe (Index BankDepth, Float)) -- V write
     , KvBank dom)
  -> ( Signal dom Float                            -- K read data
     , Signal dom Float )                          -- V read data
mapKVPorts (rdAddr, rdEn, wrAddr, wrK, wrV, bank) =
  let addrA = mux rdEn rdAddr (pure 0)
      addrB = wrAddr
      wrA   = pure Nothing
      (kA, _) = runKeyBank bank   (addrA, wrA) (addrB, wrK)
      (vA, _) = runValueBank bank (addrA, wrA) (addrB, wrV)
  in  (kA, vA)
