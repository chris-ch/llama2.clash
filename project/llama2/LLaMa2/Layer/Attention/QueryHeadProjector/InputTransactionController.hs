module LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController
  ( InputTransactionIn(..)
  , InputTransactionOut(..)
  , inputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumQueryHeads)
import qualified Prelude as P

import TraceUtils (traceEdgeC)

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
  -> Index NumQueryHeads
  -> InputTransactionIn dom
  -> InputTransactionOut dom
inputTransactionController cycleCounter headIdx inputs =
  InputTransactionOut { itcLatchedValid = latchedValidTraced }
  where
    tag = "[ITC H" P.++ show headIdx P.++ "] "

    -- Input valid latch: SET on inputValid, CLR when complete and downstream ready
    latchedValid = register False nextLatchedValid
    clearCondition = itcConsumeSignal inputs
    nextLatchedValid =
      mux (itcInputValid inputs .&&. (not <$> latchedValid)) (pure True)
      $ mux clearCondition (pure False)
        latchedValid

    latchedValidTraced = traceEdgeC cycleCounter (tag P.++ "latchedValid") latchedValid
