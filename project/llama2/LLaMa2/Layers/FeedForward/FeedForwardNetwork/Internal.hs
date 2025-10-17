module LLaMa2.Layers.FeedForward.FeedForwardNetwork.Internal  (
  feedForwardCore
)where

import Clash.Prelude
import LLaMa2.Config
    (
      ModelDimension,
      ModelDimension, HiddenDimension )

import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Numeric.Fixed
import LLaMa2.Layers.Components.Quantized (FeedForwardNetworkComponentQ (..))
import LLaMa2.Helpers.MatVecI8E (parallel32RowMatrixMultiplier)

sigmoidLinearUnit :: FixedPoint -> FixedPoint
sigmoidLinearUnit x = x / (1 + expF (negate x))
  where
    -- reuse your expF definition
    expF = LLaMa2.Numeric.Fixed.expF

-- FSM States for FFN pipeline
data FFNState = FFNIdle | FFNGate | FFNUp | FFNDown | FFNDone
  deriving (Show, Eq, Generic, NFDataX)

-- Implements W1 (gate) -> W3 (up) -> W2 (down) pipeline with proper protocol
feedForwardCore :: forall dom . HiddenClockResetEnable dom
  => Signal dom Bool                              -- ^ validIn
  -> Signal dom Bool                              -- ^ readyIn (from downstream)
  -> FeedForwardNetworkComponentQ
  -> Signal dom (Vec ModelDimension FixedPoint)   -- ^ input vector (normalized)
  -> ( Signal dom (Vec ModelDimension FixedPoint) -- ^ output vector
     , Signal dom Bool                             -- ^ validOut
     , Signal dom Bool                             -- ^ readyOut (to upstream)
     )
feedForwardCore validIn readyIn ffn xHat =
  (outputResult, validOut, readyOut)
  where
    -- State machine
    state :: Signal dom FFNState
    state = register FFNIdle nextState
    
    -- Handshake conditions
    acceptInput = (state .==. pure FFNIdle) .&&. validIn
    outputAccepted = (state .==. pure FFNDone) .&&. readyIn
    
    -- W1 (gate) computation
    gateValidIn = state .==. pure FFNGate
    gateReadyIn = pure True  -- Always ready to accept multiplier result
    (gateRaw, gateValidOut, _gateReadyOut) = 
      parallel32RowMatrixMultiplier gateValidIn gateReadyIn (fW1Q ffn) xHatLatched
    
    -- W3 (up) computation  
    upValidIn = state .==. pure FFNUp
    upReadyIn = pure True
    (upRaw, upValidOut, _upReadyOut) = 
      parallel32RowMatrixMultiplier upValidIn upReadyIn (fW3Q ffn) xHatLatched
    
    -- W2 (down) computation
    downValidIn = state .==. pure FFNDown
    downReadyIn = pure True
    (downRaw, downValidOut, _downReadyOut) = 
      parallel32RowMatrixMultiplier downValidIn downReadyIn (fW2Q ffn) gateUpLatched
    
    -- State transitions with explicit conditions
    nextState = 
      mux acceptInput
          (pure FFNGate)  -- Accept input, start W1
          (mux ((state .==. pure FFNGate) .&&. gateValidOut)
               (pure FFNUp)  -- W1 done, start W3
               (mux ((state .==. pure FFNUp) .&&. upValidOut)
                    (pure FFNDown)  -- W3 done, start W2
                    (mux ((state .==. pure FFNDown) .&&. downValidOut)
                         (pure FFNDone)  -- W2 done, output ready
                         (mux outputAccepted
                              (pure FFNIdle)  -- Output consumed, return to idle
                              state))))  -- Hold current state
    
    -- Ready/Valid outputs
    readyOut :: Signal dom Bool
    readyOut = state .==. pure FFNIdle
    
    validOut :: Signal dom Bool
    validOut = state .==. pure FFNDone
    
    -- Data path: latch input when accepted
    xHatLatched :: Signal dom (Vec ModelDimension FixedPoint)
    xHatLatched = regEn (repeat 0) acceptInput xHat
    
    -- Apply SiLU activation to gate output and latch when valid
    gateSiLU :: Signal dom (Vec HiddenDimension FixedPoint)
    gateSiLU = regEn (repeat 0) gateValidOut (map sigmoidLinearUnit <$> gateRaw)

    -- Element-wise multiplication: gate ⊙ up, computed when up becomes valid
    gateUpLatched :: Signal dom (Vec HiddenDimension FixedPoint)
    gateUpLatched = regEn (repeat 0) upValidOut (zipWith (*) <$> gateSiLU <*> upRaw)
    
    -- Latch final output when valid
    outputResult :: Signal dom (Vec ModelDimension FixedPoint)
    outputResult = regEn (repeat 0) downValidOut downRaw
