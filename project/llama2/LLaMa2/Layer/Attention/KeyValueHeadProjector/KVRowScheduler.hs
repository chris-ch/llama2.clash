module LLaMa2.Layer.Attention.KeyValueHeadProjector.KVRowScheduler
  ( KVRowSchedulerIn(..)
  , KVRowSchedulerOut(..)
  , kvRowScheduler
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig

--------------------------------------------------------------------------------
-- KVRowScheduler
-- Computes next row index value for KV projection
-- (COMBINATORIAL - no internal state, identical logic to Q's RowScheduler)
--------------------------------------------------------------------------------

data KVRowSchedulerIn dom = KVRowSchedulerIn
  { kvrsRowDone       :: Signal dom Bool                     -- Row computation complete
  , kvrsOutputValid   :: Signal dom Bool                     -- All rows complete
  , kvrsConsumeSignal :: Signal dom Bool                     -- Coordinated consume
  , kvrsCurrentIndex  :: Signal dom (Index HeadDimension)    -- Current row index (registered)
  } deriving (Generic)

newtype KVRowSchedulerOut dom
  = KVRowSchedulerOut { kvrsNextRowIndex :: Signal dom (Index HeadDimension) }
  deriving (Generic)

-- | Compute next row index
--
-- == State Transitions
--
-- @
-- rowDone && (index != maxBound) → index + 1    (increment)
-- outputValid && consumeSignal   → 0            (reset for next token)
-- otherwise                      → index        (hold)
-- @
--
kvRowScheduler :: forall dom.
  KVRowSchedulerIn dom
  -> KVRowSchedulerOut dom
kvRowScheduler inputs =
  KVRowSchedulerOut { kvrsNextRowIndex = nextRowIndex }
  where
    rowIndex = kvrsCurrentIndex inputs

    -- Compute next index (COMBINATORIAL)
    nextRowIndex =
      mux (kvrsRowDone inputs .&&. (rowIndex ./=. pure maxBound)) (rowIndex + 1)  -- Increment
      $ mux (kvrsOutputValid inputs .&&. kvrsConsumeSignal inputs) (pure 0)        -- Reset
        rowIndex                                                                    -- Hold
