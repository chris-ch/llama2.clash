module Model.Layers.TransformerLayer.Internal (
    controlOneHead
) where

import Clash.Prelude
import Model.Config (ModelDimension, HeadDimension)
import Model.Helpers.MatVecI8E (sequentialMatVecStub)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D)

-- FSM states
data FSMState = IDLE | PROJECTING | DONE
    deriving (Eq, Show, Generic, NFDataX)

-- Controller for one head's WO projection
controlOneHead ::
  forall dom .
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

    -- Start WO projection when entering state 1
    startWO = fmap ( == IDLE) state .&&. headDoneRising

    -- Call the sequential matmul stub (no latching - use current headOutput)
    (woResult, woValidOut, _woReadyOut) =
      sequentialMatVecStub woMatrix (bundle (startWO, headOutput))

    -- Output the result (hold it when valid)
    projOut = regEn (repeat 0) woValidOut woResult

    -- Valid out in DONE state (state 2)
    validOut = fmap (== DONE) state

    -- Ready when idle
    readyOut = fmap (== IDLE) state

--
-- Cleaner implementation below, but does not work, probably requires fixing handshaking at higher levels
--

-- | A simple Mealy-style state machine helper
mealyState :: (HiddenClockResetEnable dom, NFDataX s)
            => s                   -- initial state
            -> (s -> i -> s)       -- next state function
            -> Signal dom i        -- input signal
            -> Signal dom s        -- state signal
mealyState initS f = mealy (\s x -> let s' = f s x in (s', s')) initS

-- | Controller for one head's WO projection
controlOneHead' ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Vec HeadDimension FixedPoint)           -- head output
  -> Signal dom Bool                                     -- head done
  -> QArray2D ModelDimension HeadDimension               -- WO matrix
  -> ( Signal dom (Vec ModelDimension FixedPoint)       -- projected output
     , Signal dom Bool                                   -- validOut
     , Signal dom Bool                                   -- readyOut
     )
controlOneHead' headOutput headDone woMatrix = (projOut, validOut, readyOut)
  where
    -- Detect rising edge of headDone
    headDonePrev    = register False headDone
    headDoneRising  = headDone .&&. (not <$> headDonePrev)

    -- Next-state function
    nextState :: FSMState -> (Bool, Bool) -> FSMState
    nextState s (hdRising, woDone) =
      case s of
        IDLE       | hdRising -> PROJECTING
        PROJECTING | woDone   -> DONE
        DONE                   -> IDLE
        _                      -> s

    -- Bundle inputs: (headDoneRising, woValidOut)
    fsmInput = bundle (headDoneRising, woValidOut)

    -- FSM state signal
    state :: Signal dom FSMState
    state = mealyState IDLE nextState fsmInput

    -- Start WO projection when FSM enters PROJECTING
    startWO :: Signal dom Bool
    startWO = fmap (== PROJECTING) state .&&. headDoneRising

    -- Sequential matvec stub
    (woResult, woValidOut, _woReadyOut) =
        sequentialMatVecStub woMatrix (bundle (startWO, headOutput))

    -- Output result when valid
    projOut = regEn (repeat 0) woValidOut woResult

    -- Valid when matvec result is ready
    validOut = woValidOut

    -- Ready when FSM is IDLE
    readyOut = fmap (== IDLE) state
