module LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController
  ( OutputTransactionIn(..), OutputTransactionOut(..)
  , outputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, NumQueryHeads, HeadDimension)
import qualified Prelude as P
import Clash.Debug (trace)

-- | Trace output valid latch with downstream ready status
traceOutputLatchEdges :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom Bool -> Signal dom (Index HeadDimension) -> Signal dom Bool
  -> Signal dom Bool
traceOutputLatchEdges layerIdx headIdx current prev ri dsr = traced
  where
    traced = go <$> current <*> prev <*> ri <*> dsr
    go curr p ridx downReady
      | curr && not p = trace (prefix P.++ "OVL_RISE ri=" P.++ show ridx) curr
      | not curr && p = trace (prefix P.++ "OVL_FALL ri=" P.++ show ridx P.++ " dsr=" P.++ show downReady) curr
      | otherwise     = curr
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

--------------------------------------------------------------------------------
-- COMPONENT: OutputTransactionController
-- Manages output valid latch and downstream handshaking
--------------------------------------------------------------------------------
data OutputTransactionIn dom = OutputTransactionIn
  { otcAllDone       :: Signal dom Bool  -- All rows computed
  , otcConsumeSignal :: Signal dom Bool  -- Coordinated consume signal
  } deriving (Generic)

newtype OutputTransactionOut dom
  = OutputTransactionOut {otcOutputValid :: Signal dom Bool}
  deriving (Generic)

outputTransactionController :: forall dom.
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)     -- For tracing
  -> Signal dom Bool                       -- downStreamReady for tracing
  -> OutputTransactionIn dom
  -> OutputTransactionOut dom
outputTransactionController layerIdx headIdx rowIndex downStreamReady inputs =
  OutputTransactionOut { otcOutputValid = outputValidTraced }
  where
    -- Output valid latch: CLR has priority over SET (critical for handshake)
    outputValidLatch = register False nextOutputValidLatch

    nextOutputValidLatch =
      mux (outputValidLatch .&&. otcConsumeSignal inputs) (pure False)  -- CLR first
      $ mux (otcAllDone inputs) (pure True)                              -- SET second
        outputValidLatch                                                 -- HOLD

    outputValidTraced = traceOutputLatchEdges layerIdx headIdx
                          outputValidLatch (register False outputValidLatch)
                          rowIndex downStreamReady
