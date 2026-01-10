module LLaMa2.Layer.Attention.QueryHeadProjector.QueryOutputTransactionController
  ( OutputTransactionIn(..), OutputTransactionOut(..)
  , outputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, NumQueryHeads, HeadDimension)
import qualified Prelude as P

import TraceUtils (traceEdgeC)

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
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)     -- rowIndex (unused, API compat)
  -> Signal dom Bool                       -- downStreamReady (unused, API compat)
  -> OutputTransactionIn dom
  -> OutputTransactionOut dom
outputTransactionController cycleCounter layerIdx headIdx _rowIndex _downStreamReady inputs =
  OutputTransactionOut { otcOutputValid = outputValidTraced }
  where
    tag = "[OTC L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

    -- Output valid latch: CLR has priority over SET (critical for handshake)
    outputValidLatch = register False nextOutputValidLatch

    nextOutputValidLatch =
      mux (outputValidLatch .&&. otcConsumeSignal inputs) (pure False)  -- CLR first
      $ mux (otcAllDone inputs) (pure True)                              -- SET second
        outputValidLatch                                                 -- HOLD

    outputValidTraced = traceEdgeC cycleCounter (tag P.++ "outputValid") outputValidLatch
