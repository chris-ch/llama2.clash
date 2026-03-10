module LLaMa2.Layer.Attention.KeyValueHeadProjector.OutputAccumulator
  ( outputAccumulator, OutputAccumIn(..), OutputAccumOut(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types
import LLaMa2.Types.ModelConfig
import qualified Prelude as P

import TraceUtils (traceWhenC)

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
outputAccumulator cycleCounter layerIdx kvHeadIdx inputs =
  OutputAccumOut
    { oaOutput   = kvOut
    , oaOutputHC = kvOutHC
    }
  where
    tag = "[KV-OA L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

    -- DRAM result accumulator
    kvOut = register (repeat 0) nextOutput

    -- Trace result value when rowDone fires
    resultTraced = traceWhenC cycleCounter (tag P.++ "result") (oaRowDone inputs) (oaRowResult inputs)

    nextOutput = mux (oaRowDone inputs)
                     (replace <$> oaRowIndex inputs <*> resultTraced <*> kvOut)
                     kvOut

    -- HC reference accumulator
    kvOutHC = register (repeat 0) nextOutputHC

    nextOutputHC = mux (oaRowDone inputs)
                       (replace <$> oaRowIndex inputs <*> oaRowResultHC inputs <*> kvOutHC)
                       kvOutHC
