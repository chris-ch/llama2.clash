module Model.Core.Types
  ( -- State machine
    CycleStage(..)
  , ProcessingState(..)
  , IntermediateData(..)
    -- Geometry and helpers
  , BankDepth
  , BankAddress
  , CacheDepth
  , TrueDualPortRunner
  , Token
  , Temperature
  , Seed
  , HiddenDimension
  , ModelDimemsion
  , NumQueryHeads
  , NumLayers
  , NumKeyValueHeads
  , SequenceLength
  , HeadDimension
  , RotaryPositionalEmbeddingDimension
  , VocabularySize
  , SingleHeadComponent(..)
  , RotaryEncodingComponent(..)
  , EmbeddingComponent(..)
  , CArray2D(..)
  ) where

import Clash.Prelude
import qualified GHC.TypeNats
import GHC.Stack (HasCallStack)
import Model.Numeric.Types (FixedPoint)

{- 
-- model config 260K
type ModelDimemsion = 64
type HiddenDimension = 172
type NumLayers = 5
type NumQueryHeads = 8
type NumKeyValueHeads = 4
type HeadDimension  = 8
type RotaryPositionalEmbeddingDimension = 4
type VocabularySize = 512 :: Nat
type SequenceLength = 512
-}

-- model config 15M
type ModelDimemsion = 288
type HiddenDimension = 768
type NumLayers = 6
type NumQueryHeads = 6
type NumKeyValueHeads = 6
type HeadDimension  = 48
type RotaryPositionalEmbeddingDimension = 24
type VocabularySize = 32000 :: Nat
type SequenceLength = 256

{- 
-- model config 42M
type ModelDimemsion = 512
type HiddenDimension = 1376
type NumLayers = 8
type NumQueryHeads = 8
type NumKeyValueHeads = 8
type HeadDimension  = 64
type RotaryPositionalEmbeddingDimension = 32
type VocabularySize = 32000 :: Nat
type SequenthLength = 1024
 -}

{-
-- model config 110M
type ModelDimemsion = 768
type HiddenDimension = 2048
type NumLayers = 12
type NumQueryHeads = 12
type NumKeyValueHeads = 12
type HeadDimension  = 64
type RotaryPositionalEmbeddingDimension = 32
type VocabularySize = 32000 :: Nat
type SequenthLength = 1024
-}

{-
-- LLaMA 2 7B
type ModelDimemsion = 4096
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
type ModelDimemsion = 5120
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
type ModelDimemsion = 7168
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

type BankDepth   = SequenceLength GHC.TypeNats.* HeadDimension
type BankAddress = Index BankDepth

-- Global KV-cache geometry (all layers × KV heads × seq × headDim)
type CacheDepth   = NumLayers GHC.TypeNats.* NumKeyValueHeads GHC.TypeNats.* SequenceLength GHC.TypeNats.* HeadDimension

-- Dual-port RAM runner type (true dual port)
type TrueDualPortRunner dom n a =
       ( Signal dom (Index n)               -- Port A address
       , Signal dom (Maybe (Index n, a)) )  -- Port A write (optional)
    -> ( Signal dom (Index n)               -- Port B address
       , Signal dom (Maybe (Index n, a)) )  -- Port B write (optional)
    -> ( Signal dom a                       -- Port A read output
       , Signal dom a )                     -- Port B read output

-- ============================================================================
-- Multi-Cycle State Machine
-- ============================================================================

data CycleStage =
    Stage1_ProjectQKV      -- compute Q,K,V for current layer & pos
  | Stage2_WriteKV         -- write K,V(pos) to cache
  | Stage3_Attend          -- read 0..pos and attend (Q uses current pos)
  | Stage4_FeedForward     -- FFN and residual
  | Stage5_Bookkeeping     -- layer+pos bookkeeping; raises readyPulse at last layer
  deriving (Show, Eq, Enum, Bounded, Generic)

instance NFDataX CycleStage where
  rnfX :: CycleStage -> ()
  rnfX x = seq x ()
  hasUndefined :: CycleStage -> Bool
  hasUndefined _ = False
  ensureSpine :: CycleStage -> CycleStage
  ensureSpine x = x
  deepErrorX :: HasCallStack => String -> CycleStage
  deepErrorX = errorX

-- Tracks which stage, which layer, and which sequence position
-- the pipeline is currently processing.
data ProcessingState = ProcessingState
  { processingStage  :: CycleStage
  , processingLayer  :: Index NumLayers
  , sequencePosition :: Index SequenceLength
  } deriving (Show, Generic, NFDataX)

-- ============================================================================
-- Intermediate Data Storage
-- ============================================================================

-- Per-layer intermediate data vectors carried through the pipeline.
-- Updated selectively depending on cycle stage.
data IntermediateData = IntermediateData
  { inputVector       :: Vec ModelDimemsion FixedPoint
  , queryVectors      :: Vec NumQueryHeads (Vec HeadDimension FixedPoint)
  , keyVectors        :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , valueVectors      :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  , attentionOutput   :: Vec ModelDimemsion FixedPoint
  , feedForwardOutput :: Vec ModelDimemsion FixedPoint
  } deriving (Show, Generic, NFDataX, Eq)

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show, Eq)

type Token = Unsigned 32
type Temperature = FixedPoint
type Seed = Unsigned 32

-- Data definitions for LLM architecture

data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: CArray2D VocabularySize ModelDimemsion,
    rmsFinalWeight :: Vec ModelDimemsion Float
  } deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: CArray2D SequenceLength RotaryPositionalEmbeddingDimension,
    freqSin :: CArray2D SequenceLength RotaryPositionalEmbeddingDimension
  } deriving (Show, Generic, Eq)

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: CArray2D HeadDimension ModelDimemsion
  , wkHead :: CArray2D HeadDimension ModelDimemsion
  , wvHead :: CArray2D HeadDimension ModelDimemsion
  , rotary :: RotaryEncodingComponent
  } deriving (Show)
