module LLaMa2.Layer.Attention.RowScheduler
  ( RowSchedulerIn(..)
  , RowSchedulerOut(..)
  , rowScheduler
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig

--------------------------------------------------------------------------------
-- RowScheduler
-- Computes next row index value (COMBINATORIAL - no internal state)
--------------------------------------------------------------------------------
data RowSchedulerIn dom = RowSchedulerIn
  { rsRowDone       :: Signal dom Bool                     -- Row computation complete
  , rsOutputValid   :: Signal dom Bool                     -- All rows complete
  , rsConsumeSignal :: Signal dom Bool                     -- Coordinated consume from parent
  , rsCurrentIndex  :: Signal dom (Index HeadDimension)    -- Current row index (REGISTERED at top level)
  } deriving (Generic)

data RowSchedulerOut dom = RowSchedulerOut
  { rsNextRowIndex :: Signal dom (Index HeadDimension)  -- Next row index (COMBINATORIAL)
  } deriving (Generic)

rowScheduler :: forall dom.
  RowSchedulerIn dom
  -> RowSchedulerOut dom
rowScheduler inputs =
  RowSchedulerOut { rsNextRowIndex = nextRowIndex }
  where
    rowIndex = rsCurrentIndex inputs  -- Use registered value from top level

    -- Compute next index (COMBINATORIAL)
    nextRowIndex = 
      mux (rsRowDone inputs .&&. (rowIndex ./=. pure maxBound)) (rowIndex + 1)  -- Increment
      $ mux (rsOutputValid inputs .&&. rsConsumeSignal inputs) (pure 0)          -- Reset
        rowIndex                                                                  -- Hold
