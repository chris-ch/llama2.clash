module LLaMa2.Layers.TransformerLayer.Internal (
    singleHeadController
) where

import Clash.Prelude
import LLaMa2.Config (ModelDimension, HeadDimension)
import LLaMa2.Helpers.MatVecI8E (matrixMultiplierStub, matrixMultiplier)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.ParamPack (MatI8E)
import LLaMa2.Helpers (liftA4)

-- FSM states
data FSMState = IDLE | REQUESTING | PROJECTING | DONE
  deriving (Eq, Show, Generic, NFDataX)

{- Ready/Valid Handshaking Protocol:
   
   REQUESTING: We have data and are asserting woValidIn, waiting for woReadyOut
   PROJECTING: Multiplier accepted our data (handshake occurred), waiting for result
-}

singleHeadController :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool                              -- validIn (head done signal)
  -> Signal dom (Vec HeadDimension FixedPoint)    -- head vector
  -> MatI8E ModelDimension HeadDimension         -- WO matrix
  -> ( Signal dom (Vec ModelDimension FixedPoint) -- projected output
     , Signal dom Bool                             -- validOut
     , Signal dom Bool                             -- readyOut (can accept new head)
     )
singleHeadController validIn headVector woMatrix = (projOut, validOut, readyOut)
  where
    -- State machine
    state :: Signal dom FSMState
    state = register IDLE nextState
    
    -- Handshake conditions
    upstreamHandshake = validIn .&&. readyOut          -- We accept input
    multiplierRequestHandshake = woValidIn .&&. woReadyOut  -- Multiplier accepts request
    multiplierResultHandshake = woValidOut .&&. woReadyIn   -- We accept result
    
    -- State transitions
    nextState = liftA4 transition state upstreamHandshake multiplierRequestHandshake multiplierResultHandshake
    
    transition :: FSMState -> Bool -> Bool -> Bool -> FSMState
    transition IDLE upHS _ _ 
      | upHS      = REQUESTING     -- Got new input, need to send to multiplier
      | otherwise = IDLE
    
    transition REQUESTING _ reqHS _
      | reqHS     = PROJECTING     -- Multiplier accepted, now waiting for result
      | otherwise = REQUESTING     -- Keep requesting until multiplier ready
    
    transition PROJECTING _ _ resHS
      | resHS     = DONE           -- Got result
      | otherwise = PROJECTING     -- Still computing
    
    transition DONE _ _ _ = IDLE   -- Output consumed, back to idle
    
    -- Ready to accept new input only when IDLE
    readyOut :: Signal dom Bool
    readyOut = liftA2 (==) state (pure IDLE)
    
    -- Latch input vector when we accept it
    latchedVector :: Signal dom (Vec HeadDimension FixedPoint)
    latchedVector = regEn (repeat 0) upstreamHandshake headVector
    
    -- Assert woValidIn when in REQUESTING state
    woValidIn :: Signal dom Bool
    woValidIn = liftA2 (==) state (pure REQUESTING)
    
    -- Ready to accept multiplier result when in PROJECTING state
    woReadyIn :: Signal dom Bool
    woReadyIn = liftA2 (==) state (pure PROJECTING)
    
    -- Call the matrix multiplier
    (woResult, woValidOut, woReadyOut) = matrixMultiplierStub woValidIn woReadyIn woMatrix latchedVector
    
    -- Latch result when we accept it
    projOut = regEn (repeat 0) multiplierResultHandshake woResult
    
    -- Valid output when in DONE state
    validOut :: Signal dom Bool
    validOut = liftA2 (==) state (pure DONE)
