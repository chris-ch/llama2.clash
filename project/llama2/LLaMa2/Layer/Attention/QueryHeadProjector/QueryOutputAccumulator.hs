module LLaMa2.Layer.Attention.QueryHeadProjector.QueryOutputAccumulator
  ( outputAccumulator, OutputAccumIn(..), OutputAccumOut(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types ( FixedPoint )
import LLaMa2.Types.ModelConfig
    ( HeadDimension, NumLayers, NumQueryHeads )
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
  -> Index NumQueryHeads
  -> OutputAccumIn dom
  -> OutputAccumOut dom
outputAccumulator cycleCounter layerIdx headIdx inputs =
  OutputAccumOut
    { oaOutput   = qOut
    , oaOutputHC = qOutHC
    }
  where
    tag = "[OA L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

    -- DRAM result accumulator
    qOut = register (repeat 0) nextOutput

    -- Trace result value when rowDone fires
    resultTraced = traceWhenC cycleCounter (tag P.++ "result") (oaRowDone inputs) (oaRowResult inputs)

    nextOutput = mux (oaRowDone inputs)
                     (replace <$> oaRowIndex inputs <*> resultTraced <*> qOut)
                     qOut

    -- HC reference accumulator
    qOutHC = register (repeat 0) nextOutputHC

    nextOutputHC = mux (oaRowDone inputs)
                       (replace <$> oaRowIndex inputs <*> oaRowResultHC inputs <*> qOutHC)
                       qOutHC
