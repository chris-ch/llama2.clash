module LLaMa2.Layer.FeedForward.FeedForwardNetwork (
   feedForwardStage
) where

import Clash.Prelude
import LLaMa2.Numeric.FixedPoint ( rmsNormFwFix )
import LLaMa2.Types.ModelConfig 
    ( ModelDimension, ModelDimension, HiddenDimension )
import LLaMa2.Numeric.Types ( FixedPoint, FixedPoint )

import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplier)
import LLaMa2.Layer.FeedForward.Activation (sigmoidLinearUnit)
import qualified Simulation.Parameters as PARAM (FeedForwardNetworkComponentQ (..))

feedForwardStage
  :: HiddenClockResetEnable dom
  => Signal dom Bool                              -- ^ validIn
  -> Signal dom Bool                              -- ^ readyIn (from downstream)
  -> PARAM.FeedForwardNetworkComponentQ
  -> Signal dom (Vec ModelDimension FixedPoint)   -- ^ input vector
  -> ( Signal dom (Vec ModelDimension FixedPoint) -- ^ output vector
     , Signal dom Bool                             -- ^ validOut
     , Signal dom Bool                             -- ^ readyOut (to upstream)
     )
feedForwardStage validIn readyIn ffn inputVector =
  (outputVector, validOut, readyOut)
  where
    -- Pre-normalize the input (combinational)
    xHat = rmsNormFwFix <$> inputVector <*> pure (PARAM.fRMSFfnF ffn)

    -- Sequential FFN core with handshaking
    (ffnCore, coreValidOut, coreReadyOut) =
      feedForwardCore validIn readyIn ffn xHat

    -- Add residual connection when core output is valid
    -- Register the residual to align with FFN output timing
    inputVectorDelayed = regEn (repeat 0) coreValidOut inputVector
    outputVector = zipWith (+) <$> inputVectorDelayed <*> ffnCore

    -- Pass through handshaking signals
    validOut = coreValidOut
    readyOut = coreReadyOut

-- FSM States for FFN pipeline
data FFNState = FFNIdle | FFNGate | FFNUp | FFNDown | FFNDone
  deriving (Show, Eq, Generic, NFDataX)

-- Implements W1 (gate) -> W3 (up) -> W2 (down) pipeline with proper protocol
feedForwardCore :: forall dom . HiddenClockResetEnable dom
  => Signal dom Bool                              -- ^ validIn
  -> Signal dom Bool                              -- ^ readyIn (from downstream)
  -> PARAM.FeedForwardNetworkComponentQ
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
    acceptInput = (state .==. pure FFNIdle) .&&. validIn .&&. gateReadyOut
    outputAccepted = (state .==. pure FFNDone) .&&. readyIn

    -- W1 (gate) computation
    gateValidIn = (state .==. pure FFNGate) .&&. gateReadyOut
    gateReadyIn = (state .==. pure FFNGate) .&&. readyIn .&&. upReadyOut
    (gateRaw, gateValidOut, gateReadyOut) =
      parallelRowMatrixMultiplier gateValidIn gateReadyIn (PARAM.fW1Q ffn) xHatLatched

    -- W3 (up) computation
    upValidIn = (state .==. pure FFNUp) .&&. upReadyOut
    upReadyIn = (state .==. pure FFNUp) .&&. readyIn .&&. downReadyOut
    (upRaw, upValidOut, upReadyOut) =
      parallelRowMatrixMultiplier upValidIn upReadyIn (PARAM.fW3Q ffn) xHatLatched

    -- W2 (down) computation
    downValidIn = (state .==. pure FFNDown) .&&. downReadyOut
    downReadyIn = (state .==. pure FFNDown) .&&. readyIn
    (downRaw, downValidOut, downReadyOut) =
      parallelRowMatrixMultiplier downValidIn downReadyIn (PARAM.fW2Q ffn) gateUpLatched

    -- State transitions with explicit conditions
    nextState =
      mux acceptInput
          (pure FFNGate)  -- Accept input, start W1
          (mux ((state .==. pure FFNGate) .&&. gateValidOut .&&. upReadyOut)
               (pure FFNUp)  -- W1 done, W3 ready
               (mux ((state .==. pure FFNUp) .&&. upValidOut .&&. downReadyOut)
                    (pure FFNDown)  -- W3 done, W2 ready
                    (mux ((state .==. pure FFNDown) .&&. downValidOut)
                         (pure FFNDone)  -- W2 done, output ready
                         (mux outputAccepted
                              (pure FFNIdle)  -- Output consumed, return to idle
                              state))))  -- Hold current state

    -- Ready/Valid outputs
    readyOut :: Signal dom Bool
    readyOut = (state .==. pure FFNIdle) .&&. gateReadyOut

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
