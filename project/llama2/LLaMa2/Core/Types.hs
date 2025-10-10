module LLaMa2.Core.Types
  ( -- State machine
    CycleStage (..),
    ProcessingState (..),
    LayerData (..),
    TrueDualPortRunner,
    Token,
    Temperature,
    Seed,
    SingleHeadComponent (..),
    RotaryEncodingComponent (..),
    EmbeddingComponent (..),
    CArray2D (..),
  )
where

import Clash.Prelude
import GHC.Stack (HasCallStack)
import LLaMa2.Config
  ( HeadDimension,
    LLaMa2Dimension,
    NumKeyValueHeads,
    NumLayers,
    NumQueryHeads,
    RotaryPositionalEmbeddingDimension,
    SequenceLength,
    VocabularySize,
  )
import LLaMa2.Numeric.Types (FixedPoint)

-- Dual-port RAM runner type (true dual port)
type TrueDualPortRunner dom n a =
  ( Signal dom (Index n), -- Port A address
    Signal dom (Maybe (Index n, a)) -- Port A write (optional)
  ) ->
  ( Signal dom (Index n), -- Port B address
    Signal dom (Maybe (Index n, a)) -- Port B write (optional)
  ) ->
  ( Signal dom a, -- Port A read output
    Signal dom a -- Port B read output
  )

-- ============================================================================
-- Multi-Cycle State Machine
-- ============================================================================

data CycleStage
  = Stage1_ProjectQKV -- compute Q,K,V for current layer & pos
  | Stage2_WriteKV -- write K,V(pos) to cache
  | Stage3_Attend -- read 0..pos and attend (Q uses current pos)
  | Stage4_FeedForward -- FFN and residual
  | Stage5_Bookkeeping -- layer+pos bookkeeping; raises readyPulse at last layer
  deriving (Show, Eq, Enum, Bounded, Generic)

instance NFDataX CycleStage where
  rnfX :: CycleStage -> ()
  rnfX x = seq x ()
  hasUndefined :: CycleStage -> Bool
  hasUndefined _ = False
  ensureSpine :: CycleStage -> CycleStage
  ensureSpine x = x
  deepErrorX :: (HasCallStack) => String -> CycleStage
  deepErrorX = errorX

-- Tracks which stage, which layer, and which sequence position
-- the pipeline is currently processing.
data ProcessingState = ProcessingState
  { processingStage :: CycleStage,
    processingLayer :: Index NumLayers,
    sequencePosition :: Index SequenceLength
  }
  deriving (Show, Generic, NFDataX)

-- ============================================================================
-- Intermediate Data Storage
-- ============================================================================

-- Per-layer intermediate data vectors carried through the pipeline.
-- Updated selectively depending on cycle stage.
data LayerData = LayerData
  { inputVector :: Vec LLaMa2Dimension FixedPoint,
    queryVectors :: Vec NumQueryHeads (Vec HeadDimension FixedPoint),
    keyVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint),
    valueVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint),
    attentionOutput :: Vec LLaMa2Dimension FixedPoint,
    feedForwardOutput :: Vec LLaMa2Dimension FixedPoint
  }
  deriving (Show, Generic, NFDataX, Eq)

newtype CArray2D (n :: Nat) (m :: Nat) = CArray2D (Vec n (Vec m Float)) deriving (Show, Eq)

type Token = Unsigned 32

type Temperature = FixedPoint

type Seed = Unsigned 32

-- Data definitions for LLM architecture

data EmbeddingComponent = EmbeddingComponent
  { vocabulary :: CArray2D VocabularySize LLaMa2Dimension,
    rmsFinalWeight :: Vec LLaMa2Dimension Float
  }
  deriving (Show)

data RotaryEncodingComponent = RotaryEncodingComponent
  { freqCos :: CArray2D SequenceLength RotaryPositionalEmbeddingDimension,
    freqSin :: CArray2D SequenceLength RotaryPositionalEmbeddingDimension
  }
  deriving (Show, Generic, Eq)

data SingleHeadComponent = SingleHeadComponent
  { wqHead :: CArray2D HeadDimension LLaMa2Dimension,
    wkHead :: CArray2D HeadDimension LLaMa2Dimension,
    wvHead :: CArray2D HeadDimension LLaMa2Dimension,
    rotary :: RotaryEncodingComponent
  }
  deriving (Show)
