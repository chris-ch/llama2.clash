module Model.Layers.TransformerLayer.Internal (
    controlOneHead
) where

import Clash.Prelude
import Model.Config (ModelDimension, HeadDimension)
import Model.Helpers.MatVecI8E (matrixMultiplierStub, matrixMultiplier)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D)

-- FSM states
data FSMState = IDLE | PROJECTING | DONE
    deriving (Eq, Show, Generic, NFDataX)

-- Controller for one head's WO projection
controlOneHead :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Vec HeadDimension FixedPoint)           -- head output
  -> Signal dom Bool                                      -- head done
  -> QArray2D ModelDimension HeadDimension               -- WO matrix
  -> ( Signal dom (Vec ModelDimension FixedPoint)        -- projected output
     , Signal dom Bool                                    -- validOut
     , Signal dom Bool                                    -- readyOut
     )
controlOneHead headOutput headDone woMatrix = (projOut, validOut, readyOut)
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
    (woResult
      , woValidOut
      , woReadyOut
       , _, _, _, _, _, _, _, _, _, _, _
       ) = matrixMultiplier woMatrix startWO headOutput

    -- Start WO projection when entering state 1
    startWO = fmap ( == IDLE) state .&&. headDoneRising .&&. woReadyOut

    -- Output the result (hold it when valid)
    projOut = regEn (repeat 0) woValidOut woResult

    -- Valid out in DONE state (state 2)
    validOut = fmap (== DONE) state

    -- Ready when idle
    readyOut = fmap (== IDLE) state
