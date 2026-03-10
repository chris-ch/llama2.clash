module LLaMa2.Memory.KVCacheLayout
  ( kvCacheKAddress
  , kvCacheVAddress
  , kvCacheTotalWords
  , kvRowBytes
  , KvCacheWords
  ) where

import Clash.Prelude
import qualified GHC.TypeNats as TN
import LLaMa2.Types.ModelConfig
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec, wordsPerFixedPointVec)

-- | Type-level total number of 64-byte words in the KV cache DRAM.
type KvCacheWords =
  2 TN.* NumLayers TN.* NumKeyValueHeads TN.* SequenceLength TN.* WordsPerFPVec HeadDimension

-- | Bytes per KV cache row (one Vec HeadDimension FixedPoint).
-- FixedPoint is 4 bytes; 16 fit per 64-byte AXI word.
kvRowBytes :: Int
kvRowBytes = wordsPerFixedPointVec @HeadDimension * 64

-- | Base address for K[layerIdx][kvHeadIdx] bank (byte address in KV DRAM).
kBankBase :: Index NumLayers -> Index NumKeyValueHeads -> Int
kBankBase layerIdx kvHeadIdx =
  (fromEnum layerIdx * natToNum @NumKeyValueHeads + fromEnum kvHeadIdx)
  * natToNum @SequenceLength
  * kvRowBytes

-- | Base address for V[layerIdx][kvHeadIdx] bank (byte address in KV DRAM).
-- V section follows all K banks.
vSectionOffset :: Int
vSectionOffset =
  natToNum @NumLayers * natToNum @NumKeyValueHeads
  * natToNum @SequenceLength * kvRowBytes

vBankBase :: Index NumLayers -> Index NumKeyValueHeads -> Int
vBankBase layerIdx kvHeadIdx =
  vSectionOffset + kBankBase layerIdx kvHeadIdx

-- | Byte address of K[layerIdx][kvHeadIdx][seqPos] in KV DRAM.
kvCacheKAddress :: Index NumLayers -> Index NumKeyValueHeads -> Index SequenceLength -> Unsigned 32
kvCacheKAddress layerIdx kvHeadIdx seqPos =
  fromIntegral $ kBankBase layerIdx kvHeadIdx + fromEnum seqPos * kvRowBytes

-- | Byte address of V[layerIdx][kvHeadIdx][seqPos] in KV DRAM.
kvCacheVAddress :: Index NumLayers -> Index NumKeyValueHeads -> Index SequenceLength -> Unsigned 32
kvCacheVAddress layerIdx kvHeadIdx seqPos =
  fromIntegral $ vBankBase layerIdx kvHeadIdx + fromEnum seqPos * kvRowBytes

-- | Total 64-byte words needed for the KV cache DRAM
-- (K section + V section, both indexed by [layers][kvHeads][seqPos]).
kvCacheTotalWords :: Int
kvCacheTotalWords =
  2 * natToNum @NumLayers * natToNum @NumKeyValueHeads
  * natToNum @SequenceLength
  * wordsPerFixedPointVec @HeadDimension
