module Simulation.MatVecSim
  ( matrixVectorMult
  , matrixMultiplierStub
  , matrixMultiplier'
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

-- Ready/Valid sequential façade for matrix-vector multiplication (STUB)
-- This stub adds proper ready/valid handshaking even though computation is combinational
matrixMultiplierStub
 :: forall dom rows cols .
 ( HiddenClockResetEnable dom
 , KnownNat cols, KnownNat rows
 )
 => Signal dom Bool -- ^ validIn input from the upstream producer, indicating that the input vector is valid.
 -> Signal dom Bool -- ^ readyIn from downstream consumer, indicating it can accept new data
 -> MatI8E rows cols -- ^ matrix as row vectors.
 -> Signal dom (Vec cols FixedPoint) -- ^ input vector.
 -> ( Signal dom (Vec rows FixedPoint) -- ^ output vector.
 , Signal dom Bool -- ^ validOut indicating downstream consumer that the output vector is valid.
 , Signal dom Bool -- ^ readyOut input to the producer, indicating that the multiplier is ready to accept new input.
 )
matrixMultiplierStub validIn readyIn rowsQ vecIn = (outVec, validOut, readyOut)
 where
  -- Combinational result
  resultComb = matrixVectorMult rowsQ <$> vecIn
  
  -- State: busy when we have valid output waiting for downstream
  busy = register False nextBusy
  
  -- Accept new input when not busy and validIn is high
  acceptInput = validIn .&&. (not <$> busy)
  
  -- Output consumed when busy and downstream is ready
  outputConsumed = busy .&&. readyIn
  
  -- Next busy state: become busy on new input, stay busy until consumed
  nextBusy = mux acceptInput
                 (pure True)                    -- become busy on new input
                 (mux outputConsumed
                      (pure False)              -- become idle when output consumed
                      busy)                     -- otherwise maintain state
  
  -- Latch output when we accept input
  outVec :: Signal dom (Vec rows FixedPoint)
  outVec = regEn (repeat 0) acceptInput resultComb
  
  -- Valid out: high when we have data waiting for downstream
  validOut :: Signal dom Bool
  validOut = busy
  
  -- Ready out: can accept new input when not busy
  readyOut :: Signal dom Bool
  readyOut = not <$> busy

-- | Sequential matrix-vector multiplication processor
-- | The `matrixMultiplier` function implements a sequential matrix-vector multiplication processor
-- using a handshake protocol with ready/valid signals for data flow control. It takes a matrix
-- (`MatI8E rows cols`) and an input vector (`Vec cols FixedPoint`) to compute the resulting vector
-- (`Vec rows FixedPoint`). The function operates in a clocked domain with hidden clock, reset, and
-- enable signals.
--
-- Handshaking via ready/valid signals
--
--          | ----validIn---> |            | ----validOut---> |
-- Upstream |                 | Multiplier |                  | Downstream
--          | <---readyOut--- |            | <---readyIn----- |
--
--
-- Inputs:
--   - `validIn`: A signal indicating when input data is valid (typically pulsed high for one cycle).
--   - `readyInDownstream`: A signal from the downstream module indicating it is ready to accept output.
--   - `rowVectors`: The input matrix, represented as a vector of rows (`MatI8E rows cols`).
--   - `inputVector`: A signal carrying the input vector (`Vec cols FixedPoint`) for multiplication.
--
-- Outputs:
--   - A tuple containing:
--     - `outputVector`: A signal carrying the result vector (`Vec rows FixedPoint`).
--     - `validOut`: A signal indicating when the output vector is valid.
--     - `readyOut`: A signal indicating when the multiplier is ready to accept new input.
--
-- Usage:
--   - Ensure the matrix (`rowVectors`) and input vector (`inputVector`) are correctly sized according
--     to the type constraints (`KnownNat rows` and `KnownNat cols`).
--   - Drive `validIn` high for one cycle to initiate a transaction when `readyOut` is high.
--   - Monitor `validOut` to determine when the output vector is ready, and ensure `readyInDownstream`
--     is high to allow the state machine to return to the ready state after completion.
--   - The multiplication is performed sequentially, row by row, with results accumulated in
--     `outputVector`. The handshake protocol ensures proper synchronization with upstream and
--     downstream modules.
--
-- Example:
--   For a 3x4 matrix and a 4-element input vector, the function computes the dot product for each
--   row, producing a 3-element result vector. The test case demonstrates this with a matrix
--   [[2,1,3,2], [1,2,1,3], [3,2,2,1]] and vector [2.0,1.5,1.0,0.5], yielding [9.5,7.5,11.5].
--
-- Notes:
--   - The function assumes a single-row processor (`singleRowProcessor`) handles individual row
--     multiplications, and a state machine (`matrixMultiplierStateMachine`) manages the protocol.
--   - The row index is incremented after each row computation, resetting to 0 after the final row
--     when the downstream module is ready.
--   - Use in a simulation environment (e.g., Clash) with appropriate clock and reset signals.
matrixMultiplier' :: forall dom rows cols .
     ( HiddenClockResetEnable dom
     , KnownNat cols, KnownNat rows
     )
  => Signal dom Bool        -- validIn
  -> Signal dom Bool        -- readyIn (downstream)
  -> MatI8E rows cols
  -> Signal dom (Vec cols FixedPoint)
  -> ( Signal dom (Vec rows FixedPoint)
     , Signal dom Bool      -- validOut
     , Signal dom Bool      -- readyOut
     )
matrixMultiplier' = Simulation.MatVecSim.matrixMultiplierStub
