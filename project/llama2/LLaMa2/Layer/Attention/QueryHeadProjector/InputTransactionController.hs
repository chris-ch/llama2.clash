module LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController
  ( InputTransactionIn(..)
  , InputTransactionOut(..)
  , inputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
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
  } deriving (Generic)

newtype InputTransactionOut dom
  = InputTransactionOut {itcLatchedValid :: Signal dom Bool}
  deriving (Generic)

inputTransactionController :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)  -- rowIndex (unused now, kept for API compat)
  -> InputTransactionIn dom
  -> InputTransactionOut dom
inputTransactionController cycleCounter layerIdx headIdx _rowIndex inputs =
  InputTransactionOut { itcLatchedValid = latchedValidTraced }
  where
    tag = "[ITC L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

    -- Input valid latch: SET on inputValid, CLR when complete and downstream ready
    latchedValid = register False nextLatchedValid
    
    nextLatchedValid =
      mux (itcInputValid inputs .&&. (not <$> latchedValid)) (pure True)
      $ mux (itcOutputValid inputs .&&. itcDownStreamReady inputs) (pure False)
        latchedValid

    latchedValidTraced = traceEdgeC cycleCounter (tag P.++ "latchedValid") latchedValid
