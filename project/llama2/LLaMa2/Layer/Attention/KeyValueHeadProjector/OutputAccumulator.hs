module LLaMa2.Layer.Attention.KeyValueHeadProjector.OutputAccumulator
  ( outputAccumulator, OutputAccumIn(..), OutputAccumOut(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types
import LLaMa2.Types.ModelConfig

--------------------------------------------------------------------------------
-- COMPONENT: OutputAccumulator
-- Accumulates row results into output vector
--------------------------------------------------------------------------------
data OutputAccumIn dom = OutputAccumIn
  { oaRowDone     :: Signal dom Bool
  , oaRowIndex    :: Signal dom (Index HeadDimension)
  , oaRowResult   :: Signal dom FixedPoint
  , oaRowResultHC :: Signal dom FixedPoint
  } deriving (Generic)

data OutputAccumOut dom = OutputAccumOut
  { oaOutput   :: Signal dom (Vec HeadDimension FixedPoint)
  , oaOutputHC :: Signal dom (Vec HeadDimension FixedPoint)
  } deriving (Generic)

outputAccumulator :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> OutputAccumIn dom
  -> OutputAccumOut dom
outputAccumulator _cycleCounter _layerIdx _kvHeadIdx inputs =
  OutputAccumOut
    { oaOutput   = kvOut
    , oaOutputHC = kvOutHC
    }
  where
    -- DRAM result accumulator
    kvOut = register (repeat 0) nextOutput

    nextOutput = mux (oaRowDone inputs)
                     (replace <$> oaRowIndex inputs <*> oaRowResult inputs <*> kvOut)
                     kvOut

    -- HC reference accumulator
    kvOutHC = register (repeat 0) nextOutputHC

    nextOutputHC = mux (oaRowDone inputs)
                       (replace <$> oaRowIndex inputs <*> oaRowResultHC inputs <*> kvOutHC)
                       kvOutHC
