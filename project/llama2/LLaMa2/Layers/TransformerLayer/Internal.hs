module LLaMa2.Layers.TransformerLayer.Internal (
    singleHeadController
) where

import Clash.Prelude
import LLaMa2.Config (LLaMa2Dimension, HeadDimension)
import LLaMa2.Helpers.MatVecI8E (matrixMultiplierStub, matrixMultiplier)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.ParamPack (MatI8E)

-- FSM states
data FSMState = IDLE | PROJECTING | DONE
    deriving (Eq, Show, Generic, NFDataX)

-- Controller for one head's WO projection
singleHeadController :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Vec HeadDimension FixedPoint)            -- head vector
  -> Signal dom Bool                                      -- head done
  -> MatI8E LLaMa2Dimension HeadDimension                 -- WO matrix
  -> ( Signal dom (Vec LLaMa2Dimension FixedPoint)        -- projected output
     , Signal dom Bool                                    -- validOut
     , Signal dom Bool                                    -- readyOut
     )
singleHeadController headVector headDone woMatrix = (projOut, validOut, readyOut)
  where
    -- Detect rising edge of headDone
    headDonePrev = register False headDone
    headDoneRising = headDone .&&. (not <$> headDonePrev)

    -- State: IDLE (0) -> PROJECTING (1) -> DONE (2)
    state :: Signal dom FSMState
    state = register IDLE nextState

    nextState =
      mux (fmap ( == IDLE) state .&&. headDoneRising) (pure PROJECTING) $  -- Rising edge of headDone -> start
      mux (fmap ( == PROJECTING) state .&&. woValidOut) (pure DONE) $      -- WO done -> done state
      mux (fmap ( == DONE) state) (pure IDLE)                        -- Reset to idle next cycle
      state                                              -- Hold state

    -- Call the sequential matmul
    (woResult, woValidOut, woReadyOut) = matrixMultiplier validIn (pure True) woMatrix headVector

    -- Start WO projection when entering state 1
    validIn = fmap ( == IDLE) state .&&. headDoneRising .&&. woReadyOut

    -- Output the result (hold it when valid)
    projOut = regEn (repeat 0) woValidOut woResult

    -- Valid out in DONE state (state 2)
    validOut = fmap (== DONE) state

    -- Ready when idle
    readyOut = fmap (== IDLE) state
