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
    CacheDepth
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


{-
-- LLaMA 2 7B
type ModelDimension = 4096
type HiddenDimension = 11008
type NumLayers = 32
type NumQueryHeads = 32
type NumKeyValueHeads = 32
type HeadDimension  = 128
type RotaryPositionalEmbeddingDimension = 128
type VocabularySize = 32000 :: Nat
type SequenceLength = 4096
-}

{-
-- LLaMA 2 13B
type ModelDimension = 5120
type HiddenDimension = 13824
type NumLayers = 40
type NumQueryHeads = 40
type NumKeyValueHeads = 40
type HeadDimension  = 128
type RotaryPositionalEmbeddingDimension = 128
type VocabularySize = 32000 :: Nat
type SequenceLength = 4096
-}

{-
-- LLaMA 2 70B
type ModelDimension = 7168
type HiddenDimension = 28672
type NumLayers = 70
type NumQueryHeads = 64
type NumKeyValueHeads = 64
type HeadDimension  = 112
type RotaryPositionalEmbeddingDimension = 256
type VocabularySize = 32000 :: Nat
type SequenceLength = 4096
-}

-- ============================================================================
-- Bank and Cache Geometry
-- ============================================================================

type BankDepth   = SequenceLength TN.* HeadDimension
type BankAddress = Index BankDepth

-- Global KV-cache geometry (all layers × KV heads × seq × headDim)
type CacheDepth   = NumLayers TN.* NumKeyValueHeads TN.* SequenceLength TN.* HeadDimension
