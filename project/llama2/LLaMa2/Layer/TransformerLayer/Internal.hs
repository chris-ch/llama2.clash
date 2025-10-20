module LLaMa2.Layer.TransformerLayer.Internal (
    singleHeadController
) where

import Clash.Prelude
import LLaMa2.Config (ModelDimension, HeadDimension)
import LLaMa2.Helpers.MatVecI8E (matrixMultiplier)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.ParamPack (MatI8E)

-- FSM states
data FSMState = IDLE | REQUESTING | PROJECTING | DONE
  deriving (Eq, Show, Generic, NFDataX)

{-|
  Ready/Valid Handshaking Protocol with internal backpressure control:

  • IDLE: Waiting for new input (validIn && readyOut)
  • REQUESTING: Sending request to multiplier until it accepts (woReadyOut)
  • PROJECTING: Multiplier is computing. We assert woReadyIn based on downstream readiness.
  • DONE: Output is valid, waiting to be consumed.
-}
singleHeadController :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool                              -- validIn (head done signal)
  -> Signal dom (Vec HeadDimension FixedPoint)    -- head vector
  -> MatI8E ModelDimension HeadDimension          -- WO matrix
  -> ( Signal dom (Vec ModelDimension FixedPoint) -- projected output
     , Signal dom Bool                            -- validOut
     , Signal dom Bool                            -- readyOut (can accept new head)
     )
singleHeadController validIn headVector woMatrix = (projOut, validOut, readyOut)
  where
    -- === FSM State ===
    state :: Signal dom FSMState
    state = register IDLE nextState

    -- === Handshakes ===
    upstreamHandshake         = validIn .&&. readyOut
    multiplierRequestHandshake = woValidIn .&&. woReadyOut
    multiplierResultHandshake  = woValidOut .&&. internalReady  -- now uses internal backpressure

    -- === State Transition Logic ===
    nextState = transition <$> state 
                <*> upstreamHandshake 
                <*> multiplierRequestHandshake 
                <*> multiplierResultHandshake

    transition :: FSMState -> Bool -> Bool -> Bool -> FSMState
    transition IDLE upHS _ _ 
      | upHS      = REQUESTING
      | otherwise = IDLE

    transition REQUESTING _ reqHS _
      | reqHS     = PROJECTING
      | otherwise = REQUESTING

    transition PROJECTING _ _ resHS
      | resHS     = DONE
      | otherwise = PROJECTING

    transition DONE _ _ _ = IDLE

    -- === Ready signals ===
    readyOut :: Signal dom Bool
    readyOut = (==) <$> state <*> pure IDLE

    -- === Input latch ===
    latchedVector :: Signal dom (Vec HeadDimension FixedPoint)
    latchedVector = regEn (repeat 0) upstreamHandshake headVector

    -- === Multiplier interface ===
    woValidIn :: Signal dom Bool
    woValidIn = (==) <$> state <*> pure REQUESTING

    -- Internal backpressure: allow multiplier progress if computing OR consumer ready
    internalReady :: Signal dom Bool
    internalReady = mux (state .==. pure PROJECTING)
                        (pure True)   -- while computing, proceed freely
                        readyOut      -- when done, only accept if consumer ready

    -- Connect to matrix multiplier
    (woResult, woValidOut, woReadyOut) =
      matrixMultiplier woValidIn internalReady woMatrix latchedVector

    -- === Output latch and valid signal ===
    projOut :: Signal dom (Vec ModelDimension FixedPoint)
    projOut = regEn (repeat 0) multiplierResultHandshake woResult

    validOut :: Signal dom Bool
    validOut = (==) <$> state <*> pure DONE
