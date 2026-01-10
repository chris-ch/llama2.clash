module LLaMa2.Layer.Attention.KeyValueHeadProjector.KVOutputTransactionController
  ( KVOutputTransactionIn(..)
  , KVOutputTransactionOut(..)
  , kvOutputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import qualified Prelude as P

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- KVOutputTransactionController
-- Manages output valid latch and downstream handshaking for KV heads
-- (Same logic as Q's OutputTransactionController, different type signature)
--------------------------------------------------------------------------------

data KVOutputTransactionIn dom = KVOutputTransactionIn
  { kvotcAllDone       :: Signal dom Bool  -- All rows computed (K and V)
  , kvotcConsumeSignal :: Signal dom Bool  -- Coordinated consume signal
  } deriving (Generic)

newtype KVOutputTransactionOut dom
  = KVOutputTransactionOut { kvotcOutputValid :: Signal dom Bool }
  deriving (Generic)

kvOutputTransactionController :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumKeyValueHeads            -- Note: KV head index
  -> Signal dom (Index HeadDimension)   -- rowIndex (API compat)
  -> Signal dom Bool                     -- downStreamReady (API compat)
  -> KVOutputTransactionIn dom
  -> KVOutputTransactionOut dom
kvOutputTransactionController cycleCounter layerIdx kvHeadIdx _rowIndex _downStreamReady inputs =
  KVOutputTransactionOut { kvotcOutputValid = outputValidTraced }
  where
    tag = "[KVOTC L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

    -- Output valid latch: CLR has priority over SET (critical for handshake)
    outputValidLatch = register False nextOutputValidLatch

    nextOutputValidLatch =
      mux (outputValidLatch .&&. kvotcConsumeSignal inputs) (pure False)  -- CLR first
      $ mux (kvotcAllDone inputs) (pure True)                              -- SET second
        outputValidLatch                                                   -- HOLD

    outputValidTraced = traceEdgeC cycleCounter (tag P.++ "outputValid") outputValidLatch
