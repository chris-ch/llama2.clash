module Model.Memory.Addressing
  ( computeBankAddress
  , computeExpAddress
  , dimFromBankAddress
  ) where

import Clash.Prelude
import Model.Config (BankAddress, SequenceLength, HeadDimension, BankDepth)
import Model.Config.KVGroups (KVExpAddress, KVExpGroups, KVExpGroupLg2)

-- Bank address = time * HeadDimension + headDimIndex
computeBankAddress :: Index SequenceLength -> Index HeadDimension -> BankAddress
computeBankAddress t d =
  toEnum (fromEnum d + fromEnum t * natToNum @HeadDimension)

-- Given a bank address, recover head-dimension index (rdAddr mod HeadDimension)
dimFromBankAddress :: Index BankDepth -> Index HeadDimension
dimFromBankAddress addr =
  let hd = natToNum @HeadDimension
  in toEnum (fromEnum addr `mod` hd)

-- Exponent address groups HeadDimension by 2^KVExpGroupLg2.
-- expAddr = t * KVExpGroups + groupIdx(d)
computeExpAddress :: Index SequenceLength -> Index HeadDimension -> KVExpAddress
computeExpAddress t d =
  let g = natToNum @KVExpGroups
      grp = fromEnum d `shiftR` natToNum @KVExpGroupLg2
  in toEnum (fromEnum t * g + grp)
