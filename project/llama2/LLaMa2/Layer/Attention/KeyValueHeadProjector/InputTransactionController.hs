module LLaMa2.Layer.Attention.KeyValueHeadProjector.InputTransactionController
  ( InputTransactionIn(..)
  , InputTransactionOut(..)
  , inputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig

--------------------------------------------------------------------------------
-- InputTransactionController
-- Manages input valid latch - captures inputValid and holds until completion
--------------------------------------------------------------------------------
data InputTransactionIn dom = InputTransactionIn
  { itcInputValid      :: Signal dom Bool  -- External input valid strobe
  , itcOutputValid     :: Signal dom Bool  -- Computation complete signal
  , itcDownStreamReady :: Signal dom Bool  -- Downstream consumer ready
  , itcConsumeSignal   :: Signal dom Bool
  } deriving (Generic)

newtype InputTransactionOut dom
  = InputTransactionOut {itcLatchedValid :: Signal dom Bool}
  deriving (Generic)

inputTransactionController :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom (Index HeadDimension)  -- rowIndex (unused now, kept for API compat)
  -> InputTransactionIn dom
  -> InputTransactionOut dom
inputTransactionController _cycleCounter _layerIdx _kvHeadIdx _rowIndex inputs =
  InputTransactionOut { itcLatchedValid = latchedValid }
  where
    -- Input valid latch: SET on inputValid, CLR when complete and downstream ready
    latchedValid = register False nextLatchedValid
    clearCondition = itcConsumeSignal inputs
    nextLatchedValid =
      mux (itcInputValid inputs .&&. (not <$> latchedValid)) (pure True)
      $ mux clearCondition (pure False)
        latchedValid
