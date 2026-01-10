module LLaMa2.Layer.Attention.KeyValueHeadProjector.KeyValueInputTransactionController
  ( KVInputTransactionIn(..)
  , KVInputTransactionOut(..)
  , kvInputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import qualified Prelude as P

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- KVInputTransactionController
-- Manages input valid latch for KV heads
-- (Same logic as Q's InputTransactionController, different type signature)
--------------------------------------------------------------------------------

data KVInputTransactionIn dom = KVInputTransactionIn
  { kvitcInputValid      :: Signal dom Bool  -- External input valid strobe
  , kvitcOutputValid     :: Signal dom Bool  -- Computation complete signal
  , kvitcDownStreamReady :: Signal dom Bool  -- Downstream consumer ready
  , kvitcConsumeSignal   :: Signal dom Bool  -- Coordinated consume signal
  } deriving (Generic)

newtype KVInputTransactionOut dom
  = KVInputTransactionOut { kvitcLatchedValid :: Signal dom Bool }
  deriving (Generic)

kvInputTransactionController :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumKeyValueHeads           -- Note: KV head index
  -> Signal dom (Index HeadDimension)  -- rowIndex (API compat)
  -> KVInputTransactionIn dom
  -> KVInputTransactionOut dom
kvInputTransactionController cycleCounter layerIdx kvHeadIdx _rowIndex inputs =
  KVInputTransactionOut { kvitcLatchedValid = latchedValidTraced }
  where
    tag = "[KVITC L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

    -- Input valid latch: SET on inputValid, CLR when complete and downstream ready
    latchedValid = register False nextLatchedValid
    clearCondition = kvitcConsumeSignal inputs
    
    nextLatchedValid =
      mux (kvitcInputValid inputs .&&. (not <$> latchedValid)) (pure True)
      $ mux clearCondition (pure False)
        latchedValid

    latchedValidTraced = traceEdgeC cycleCounter (tag P.++ "latchedValid") latchedValid
