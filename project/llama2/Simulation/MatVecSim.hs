module Simulation.MatVecSim
  ( matrixVectorMult
  , matrixMultiplierStub
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.ParamPack (MatI8E, RowI8E, dequantRowToF)
import LLaMa2.Helpers.FixedPoint (dotProductF)

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
