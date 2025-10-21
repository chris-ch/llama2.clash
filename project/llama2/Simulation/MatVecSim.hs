module Simulation.MatVecSim
  ( matrixVectorMult
  , matrixMultiplierStub
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E, dequantRowToF)
import LLaMa2.Numeric.FixedPoint (dotProductF)

-- Dot product: dequantize a row once, then reuse existing F dot-product.
dotProductRowI8E :: KnownNat n => RowI8E n -> Vec n FixedPoint -> FixedPoint
dotProductRowI8E row = dotProductF (dequantRowToF row)

-- Matrix @ vector where matrix is quantized (I8E rows) and vector is FixedPoint.
matrixVectorMult
  :: (KnownNat cols)
  => MatI8E rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMult byRows xF =
  map (`dotProductRowI8E` xF) byRows

-- State machine to emulate multi-cycle processing
data State = Idle | Processing | Done
  deriving (Generic, NFDataX, Show, Eq)

-- Ready/Valid sequential façade for matrix-vector multiplication (STUB)
-- This stub adds proper ready/valid handshaking even though computation is combinational
matrixMultiplierStub
 :: forall dom rows cols .
 ( HiddenClockResetEnable dom
 , KnownNat cols, KnownNat rows
 )
 => Signal dom Bool -- ^ validIn input from the upstream producer
 -> Signal dom Bool -- ^ readyIn from downstream consumer
 -> MatI8E rows cols -- ^ matrix as row vectors
 -> Signal dom (Vec cols FixedPoint) -- ^ input vector
 -> ( Signal dom (Vec rows FixedPoint) -- ^ output vector
    , Signal dom Bool -- ^ validOut indicating output vector is valid
    , Signal dom Bool -- ^ readyOut indicating readiness for new input
    )
matrixMultiplierStub validIn readyIn rowsQ vecIn = (outVec, validOut, readyOut)
 where
  -- Combinational result
  resultComb = matrixVectorMult rowsQ <$> vecIn

  state = register Idle nextState

  delayMax = pure 2 -- 1 cycle delay messes up the output token stream, requires at least 2

  -- Delay counter to emulate processing time
  delayCounter = register (0 :: Unsigned 16) nextDelayCounter

  -- Accept input when idle and validIn is high
  acceptInput = (state .==. pure Idle) .&&. validIn

  -- Output consumed when in Done state and downstream is ready
  outputConsumed = (state .==. pure Done) .&&. readyIn

  -- State transitions
  nextState = mux (state .==. pure Idle)
                 (mux acceptInput (pure Processing) (pure Idle))
                 (mux (state .==. pure Processing)
                      (mux (delayCounter .==. delayMax) (pure Done) (pure Processing))
                      (mux outputConsumed (pure Idle) (pure Done)))

  -- Delay counter logic
  nextDelayCounter = mux acceptInput
                         (pure 1)
                         (mux (state .==. pure Processing .&&. delayCounter .<. delayMax)
                              (delayCounter + 1)
                              delayCounter)

  -- Latch input vector when accepted
  latchedVec = regEn (repeat 0) acceptInput vecIn

  -- Compute result (combinational, but latched)
  outVec = regEn (repeat 0) (state .==. pure Processing .&&. delayCounter .==. delayMax) resultComb

  -- Valid out when in Done state
  validOut = state .==. pure Done

  -- Ready out when in Idle state
  readyOut = state .==. pure Idle
