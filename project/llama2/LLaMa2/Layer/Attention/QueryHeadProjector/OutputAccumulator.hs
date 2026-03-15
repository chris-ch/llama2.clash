module LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator
  ( outputAccumulator, OutputAccumIn(..), OutputAccumOut(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types
import LLaMa2.Types.ModelConfig (NumQueryHeads)

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
  -> Index NumQueryHeads
  -> OutputAccumIn dom numRows
  -> OutputAccumOut dom numRows
outputAccumulator _cycleCounter _headIdx inputs =
  OutputAccumOut
    { oaOutput = qOut
    }
  where
    qOut = register (repeat 0) nextOutput

    nextOutput = mux (oaRowDone inputs)
                     (replace <$> oaRowIndex inputs <*> oaRowResult inputs <*> qOut)
                     qOut
