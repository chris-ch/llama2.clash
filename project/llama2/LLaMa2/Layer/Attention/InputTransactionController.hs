module LLaMa2.Layer.Attention.InputTransactionController
  ( InputTransactionIn(..), InputTransactionOut(..)
  , inputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, NumQueryHeads, HeadDimension)
import qualified Prelude as P
import Clash.Debug (trace)

-- | Trace latch state changes (rise/fall edges)
traceLatchEdges :: Index NumLayers -> Index NumQueryHeads -> P.String
  -> Signal dom Bool -> Signal dom Bool -> Signal dom (Index HeadDimension) 
  -> Signal dom Bool
traceLatchEdges layerIdx headIdx name current prev ri = traced
  where
    traced = go <$> current <*> prev <*> ri
    go curr p ridx
      | curr && not p = trace (prefix P.++ name P.++ "_RISE ri=" P.++ show ridx) curr
      | not curr && p = trace (prefix P.++ name P.++ "_FALL ri=" P.++ show ridx) curr
      | otherwise     = curr
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

--------------------------------------------------------------------------------
-- COMPONENT: InputTransactionController
-- Manages input valid latch - captures inputValid and holds until completion
--------------------------------------------------------------------------------
data InputTransactionIn dom = InputTransactionIn
  { itcInputValid      :: Signal dom Bool  -- External input valid strobe
  , itcOutputValid     :: Signal dom Bool  -- Computation complete signal
  , itcDownStreamReady :: Signal dom Bool  -- Downstream consumer ready
  } deriving (Generic)

data InputTransactionOut dom = InputTransactionOut
  { itcLatchedValid :: Signal dom Bool  -- Latched input valid (holds until cleared)
  } deriving (Generic)

inputTransactionController :: forall dom.
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)  -- For tracing only
  -> InputTransactionIn dom
  -> InputTransactionOut dom
inputTransactionController layerIdx headIdx rowIndex inputs =
  InputTransactionOut { itcLatchedValid = latchedValidTraced }
  where
    -- Input valid latch: SET on inputValid, CLR when complete and downstream ready
    latchedValid = register False nextLatchedValid
    
    nextLatchedValid =
      mux (itcInputValid inputs .&&. (not <$> latchedValid)) (pure True)
      $ mux (itcOutputValid inputs .&&. itcDownStreamReady inputs) (pure False)
        latchedValid

    latchedValidTraced = traceLatchEdges layerIdx headIdx "IVL"
                           latchedValid (register False latchedValid) rowIndex
