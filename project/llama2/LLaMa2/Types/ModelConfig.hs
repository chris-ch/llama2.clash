{-# LANGUAGE CPP #-}
module LLaMa2.Types.ModelConfig (
    ModelDimension,
    HiddenDimension,
    NumLayers,
    NumQueryHeads,
    NumKeyValueHeads,
    HeadDimension,
    RotaryPositionalEmbeddingDimension,
    VocabularySize,
    SequenceLength,
    BankDepth,
    BankAddress,
    QHeadsPerKVBank,
    QBramPerBankDepth
) where

import Clash.Prelude
import qualified GHC.TypeNats as TN

#ifdef MODEL_260K
-- model config 260K
type ModelDimension = 64
type HiddenDimension = 172
type NumLayers = 5
type NumQueryHeads = 8
type NumKeyValueHeads = 4
type HeadDimension  = 8
type RotaryPositionalEmbeddingDimension = 4
type VocabularySize = 512
type SequenceLength = 512

#elif MODEL_15M
-- model config 15M
type ModelDimension = 288
type HiddenDimension = 768
type NumLayers = 6
type NumQueryHeads = 6
type NumKeyValueHeads = 6
type HeadDimension  = 48
type RotaryPositionalEmbeddingDimension = 24
type VocabularySize = 32000
type SequenceLength = 256

#elif MODEL_42M
-- model config 42M
type ModelDimension = 512
type HiddenDimension = 1376
type NumLayers = 8
type NumQueryHeads = 8
type NumKeyValueHeads = 8
type HeadDimension  = 64
type RotaryPositionalEmbeddingDimension = 32
type VocabularySize = 32000
type SequenceLength = 1024

#elif MODEL_110M
-- model config 110M
type ModelDimension = 768
type HiddenDimension = 2048
type NumLayers = 12
type NumQueryHeads = 12
type NumKeyValueHeads = 12
type HeadDimension  = 64
type RotaryPositionalEmbeddingDimension = 32
type VocabularySize = 32000
type SequenceLength = 1024

#elif MODEL_7B
-- model config 7B
type ModelDimension = 4096
type HiddenDimension = 11008
type NumLayers = 32
type NumQueryHeads = 32
type NumKeyValueHeads = 32
type HeadDimension  = 128
type RotaryPositionalEmbeddingDimension = 64   -- HeadDimension / 2
type VocabularySize = 32000 :: Nat
type SequenceLength = 4096

#elif MODEL_13B
-- model config 13B
type ModelDimension = 5120
type HiddenDimension = 13824
type NumLayers = 40
type NumQueryHeads = 40
type NumKeyValueHeads = 40
type HeadDimension  = 128
type RotaryPositionalEmbeddingDimension = 64   -- HeadDimension / 2
type VocabularySize = 32000 :: Nat
type SequenceLength = 4096

#elif MODEL_70B
-- model config 70B
type ModelDimension = 7168
type HiddenDimension = 28672
type NumLayers = 70
type NumQueryHeads = 64
type NumKeyValueHeads = 64
type HeadDimension  = 112
type RotaryPositionalEmbeddingDimension = 56   -- HeadDimension / 2
type VocabularySize = 32000 :: Nat
type SequenceLength = 4096

#elif MODEL_NANO
-- nano model config for fast simulation tests
-- Constraints: HeadDimension * NumQueryHeads = ModelDimension (2*4=8)
--              All dims <= 63 so WordsPerRow = 1 for everything (single-beat DRAM)
type ModelDimension = 8
type HiddenDimension = 4
type NumLayers = 2
type NumQueryHeads = 4
type NumKeyValueHeads = 2
type HeadDimension  = 2
type RotaryPositionalEmbeddingDimension = 1
type VocabularySize = 512
type SequenceLength = 512

#else
-- defaults to model config 260K
type ModelDimension = 64
type HiddenDimension = 172
type NumLayers = 5
type NumQueryHeads = 8
type NumKeyValueHeads = 4
type HeadDimension  = 8
type RotaryPositionalEmbeddingDimension = 4
type VocabularySize = 512
type SequenceLength = 512

#endif

-- ============================================================================
-- Bank and Cache Geometry
-- ============================================================================

type BankDepth   = SequenceLength TN.* HeadDimension
type BankAddress = Index BankDepth

-- Q-BRAM geometry per KV bank
type QHeadsPerKVBank   = NumQueryHeads `TN.Div` NumKeyValueHeads
type QBramPerBankDepth = QHeadsPerKVBank TN.* HeadDimension
