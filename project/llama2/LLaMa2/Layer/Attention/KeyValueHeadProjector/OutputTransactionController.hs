module LLaMa2.Layer.Attention.KeyValueHeadProjector.OutputTransactionController
  ( OutputTransactionIn(..), OutputTransactionOut(..)
  , outputTransactionController
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, NumKeyValueHeads, HeadDimension)

--------------------------------------------------------------------------------
-- COMPONENT: OutputTransactionController
-- Manages output valid latch and downstream handshaking
--------------------------------------------------------------------------------
data OutputTransactionIn dom = OutputTransactionIn
  { otcAllDone       :: Signal dom Bool  -- All rows computed (K and V)
  , otcConsumeSignal :: Signal dom Bool  -- Coordinated consume signal
  } deriving (Generic)

newtype OutputTransactionOut dom
  = OutputTransactionOut {otcOutputValid :: Signal dom Bool}
  deriving (Generic)

outputTransactionController :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom (Index HeadDimension)     -- rowIndex (unused, API compat)
  -> Signal dom Bool                       -- downStreamReady (unused, API compat)
  -> OutputTransactionIn dom
  -> OutputTransactionOut dom
outputTransactionController _cycleCounter _layerIdx _kvHeadIdx _rowIndex _downStreamReady inputs =
  OutputTransactionOut { otcOutputValid = outputValidLatch }
  where
    -- Output valid latch: CLR has priority over SET (critical for handshake)
    outputValidLatch = register False nextOutputValidLatch

    nextOutputValidLatch =
      mux (outputValidLatch .&&. otcConsumeSignal inputs) (pure False)  -- CLR first
      $ mux (otcAllDone inputs) (pure True)                              -- SET second
        outputValidLatch                                                 -- HOLD
