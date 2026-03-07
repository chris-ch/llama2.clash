module LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator
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
data OutputAccumIn dom numRows = OutputAccumIn
  { oaRowDone   :: Signal dom Bool
  , oaRowIndex  :: Signal dom (Index numRows)
  , oaRowResult :: Signal dom FixedPoint
  } deriving (Generic)

newtype OutputAccumOut dom numRows = OutputAccumOut
  { oaOutput :: Signal dom (Vec numRows FixedPoint)
  } deriving (Generic)

outputAccumulator :: forall dom numRows.
  ( HiddenClockResetEnable dom
  , KnownNat numRows
  )
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumQueryHeads
  -> OutputAccumIn dom numRows
  -> OutputAccumOut dom numRows
outputAccumulator cycleCounter layerIdx headIdx inputs =
  OutputAccumOut
    { oaOutput = qOut
    }
  where
    tag = "[OA L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

    qOut = register (repeat 0) nextOutput

    resultTraced = traceWhenC cycleCounter (tag P.++ "result") (oaRowDone inputs) (oaRowResult inputs)

    nextOutput = mux (oaRowDone inputs)
                     (replace <$> oaRowIndex inputs <*> resultTraced <*> qOut)
                     qOut
