module LLaMa2.Types.LayerData
  (
    LayerData (..),
    LayerDataAddr (..),
    NumActivationSlots,
    ActivationBramDepth,
    ActivationBramAddr,
    initialLayerDataAddr,
    Token,
    Temperature,
    Seed,
    RotaryEncodingComponentF (..)
  )
where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( HeadDimension,
    ModelDimension,
    NumKeyValueHeads,
    NumQueryHeads,
    RotaryPositionalEmbeddingDimension,
    SequenceLength,
  )
import qualified GHC.TypeNats as TN
import LLaMa2.Numeric.Types (FixedPoint)

-- ============================================================================
-- Intermediate Data Storage
-- ============================================================================

-- {- Simulation only: not used in synthesised modules -}
-- Per-layer intermediate data vectors carried through the pipeline.
-- Updated selectively depending on cycle stage.
data LayerData = LayerData
  { inputVector :: Vec ModelDimension FixedPoint,
    queryVectors :: Vec NumQueryHeads (Vec HeadDimension FixedPoint),
    keyVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint),
    valueVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint),
    attentionOutput :: Vec ModelDimension FixedPoint,
    feedForwardOutput :: Vec ModelDimension FixedPoint
  }
  deriving (Show, Generic, Eq, NFDataX, BitPack)

-- ============================================================================
-- Activation BRAM Address Types
-- ============================================================================

-- Number of static activation slots in the on-chip BRAM.
type NumActivationSlots = 4

-- Total depth: 4 slots × ModelDimension words.
type ActivationBramDepth = NumActivationSlots TN.* ModelDimension

-- Address into the flat activation BRAM.
type ActivationBramAddr = Index ActivationBramDepth

-- Slot index record: all fields are compile-time constants so Clash folds
-- them to literals, and the 451K-bit layerDataReg disappears entirely.
data LayerDataAddr = LayerDataAddr
  { inputVecSlot   :: Index NumActivationSlots  -- slot 0
  , queryVecSlot   :: Index NumActivationSlots  -- slot 1
  , attnOutputSlot :: Index NumActivationSlots  -- slot 2
  , ffnOutputSlot  :: Index NumActivationSlots  -- slot 3
  } deriving (Generic, NFDataX)

initialLayerDataAddr :: LayerDataAddr
initialLayerDataAddr = LayerDataAddr 0 1 2 3

type Token = Unsigned 32

type Seed = Unsigned 32

type Temperature = FixedPoint

-- Precomputed rotary positional encoding tables (hardware-compatible type).
data RotaryEncodingComponentF = RotaryEncodingComponentF
  { freqCosF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  , freqSinF :: Vec SequenceLength (Vec RotaryPositionalEmbeddingDimension FixedPoint)
  } deriving (Generic, NFDataX, Show, Eq)
