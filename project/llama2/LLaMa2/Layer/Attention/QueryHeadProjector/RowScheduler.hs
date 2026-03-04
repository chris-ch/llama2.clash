module LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler
  ( RowSchedulerIn(..)
  , RowSchedulerOut(..)
  , rowScheduler
  ) where

import Clash.Prelude

--------------------------------------------------------------------------------
-- RowScheduler
-- Computes next row index value (COMBINATORIAL - no internal state)
--------------------------------------------------------------------------------
data RowSchedulerIn dom numRows = RowSchedulerIn
  { rsRowDone       :: Signal dom Bool
  , rsOutputValid   :: Signal dom Bool
  , rsConsumeSignal :: Signal dom Bool
  , rsCurrentIndex  :: Signal dom (Index numRows)  -- Current row index (REGISTERED at top level)
  } deriving (Generic)

newtype RowSchedulerOut dom numRows
  = RowSchedulerOut {rsNextRowIndex :: Signal dom (Index numRows)}  -- Next row index (COMBINATORIAL)
  deriving (Generic)

rowScheduler :: forall dom numRows. KnownNat numRows
  => RowSchedulerIn dom numRows
  -> RowSchedulerOut dom numRows
rowScheduler inputs =
  RowSchedulerOut { rsNextRowIndex = nextRowIndex }
  where
    rowIndex = rsCurrentIndex inputs  -- Use registered value from top level

    -- Compute next index (COMBINATORIAL)
    nextRowIndex =
      mux (rsRowDone inputs .&&. (rowIndex ./=. pure maxBound)) (rowIndex + 1)  -- Increment
      $ mux (rsOutputValid inputs .&&. rsConsumeSignal inputs) (pure 0)          -- Reset
        rowIndex                                                                  -- Hold
