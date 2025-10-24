module LLaMa2.Layer.FeedForward.FeedForwardNetworkWithRAM
  ( feedForwardStageWithRAM ) where

import Clash.Prelude
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Types.ModelConfig (ModelDimension, HiddenDimension)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E, dequantRowToF)
import LLaMa2.Types.Parameters (FeedForwardNetworkComponentQ(..))
import LLaMa2.Numeric.Operations (parallel64RowProcessor, matrixMultiplierStateMachine, MultiplierState (..))
import LLaMa2.Layer.FeedForward.Activation (sigmoidLinearUnit)

-- simple dynamic multiplier (matrix as Signal)
parallelRowMatrixMultiplierDyn :: forall dom rows cols.
  (HiddenClockResetEnable dom, KnownNat rows, KnownNat cols)
  => Signal dom Bool                       -- ^ validIn
  -> Signal dom Bool                       -- ^ readyIn (downstream)
  -> Signal dom (MatI8E rows cols)         -- ^ matrix (runtime)
  -> Signal dom (Vec cols FixedPoint)      -- ^ input vector
  -> ( Signal dom (Vec rows FixedPoint)    -- ^ output vector
     , Signal dom Bool                     -- ^ validOut
     , Signal dom Bool                     -- ^ readyOut
     )
parallelRowMatrixMultiplierDyn validIn readyIn matSig vecSig =
  (outVec, validOut, readyOut)
 where
  rowIndex :: Signal dom (Index rows)
  rowIndex = register 0 nextRow

  currentRow = (!!) <$> matSig <*> rowIndex

  (rowRes, rowDone) =
    LLaMa2.Numeric.Operations.parallel64RowProcessor rowReset rowEnable currentRow vecSig

  (state, rowReset, rowEnable, validOut, readyOut) =
    LLaMa2.Numeric.Operations.matrixMultiplierStateMachine validIn readyIn rowDone rowIndex

  -- PROVEN sequencing: only increment on rowDone and not last row; reset to 0 when MDone is consumed
  nextRow =
    mux (rowDone .&&. (rowIndex ./=. pure maxBound))
        (rowIndex + 1)
        (mux ((state .==. pure LLaMa2.Numeric.Operations.MDone) .&&. readyIn)
             (pure 0)
             rowIndex)

  outVec = register (repeat 0) nextOut
  nextOut = mux rowDone
                 (replace <$> rowIndex <*> rowRes <*> outVec)
                 outVec

feedForwardStageWithRAM ::
  HiddenClockResetEnable dom =>
  Signal dom Bool                              ->  -- validIn
  Signal dom Bool                              ->  -- readyIn (downstream)
  FeedForwardNetworkComponentQ                 ->  -- constants fallback
  Signal dom (MatI8E HiddenDimension ModelDimension) ->  -- RAM W1
  Signal dom (MatI8E ModelDimension HiddenDimension) ->  -- RAM W2
  Signal dom (MatI8E HiddenDimension ModelDimension) ->  -- RAM W3
  Signal dom (RowI8E ModelDimension)           ->  -- RAM rmsFfn
  Signal dom Bool                              ->  -- useRAMFFN
  Signal dom (Vec ModelDimension FixedPoint)   ->  -- input vector (pre-normalized)
  ( Signal dom (Vec ModelDimension FixedPoint)
  , Signal dom Bool
  , Signal dom Bool )
feedForwardStageWithRAM validIn readyIn ffn ramW1 ramW2 ramW3 ramRms useRAMFFN xIn =
  (outVec, validOut, readyOut)
 where
  letW1 = mux useRAMFFN ramW1 (pure (fW1Q ffn))
  letW2 = mux useRAMFFN ramW2 (pure (fW2Q ffn))
  letW3 = mux useRAMFFN ramW3 (pure (fW3Q ffn))
  
  -- Normalize with RAM rms when enabled (using dequantRowToF)
  xHat = rmsNormFwFix 
    <$> xIn
    <*> mux useRAMFFN (dequantRowToF <$> ramRms) (pure (fRMSFfnF ffn))
  
  -- Rest of the code remains unchanged...
  state = register (0::Unsigned 3) (mux accept (pure 1) (mux (state .==. pure 1 .&&. gateV) (pure 2)
                         (mux (state .==. pure 2 .&&. upV) (pure 3)
                         (mux (state .==. pure 3 .&&. downV) (pure 0) state))))
  accept = (state .==. pure 0) .&&. validIn
  gateValidIn = state .==. pure 1
  upValidIn   = state .==. pure 2
  downValidIn = state .==. pure 3
  (gateRaw, gateV, _gr) = parallelRowMatrixMultiplierDyn gateValidIn (pure True) letW1 xHat
  (upRaw,   upV,   _ur) = parallelRowMatrixMultiplierDyn upValidIn   (pure True) letW3 xHat
  gateSiLU = regEn (repeat 0) gateV (map sigmoidLinearUnit <$> gateRaw)
  gateUp   = regEn (repeat 0) upV   (zipWith (*) <$> gateSiLU <*> upRaw)
  (downRaw, downV, _dr) = parallelRowMatrixMultiplierDyn downValidIn (pure True) letW2 gateUp
  outVec   = regEn (repeat 0) downV downRaw
  validOut = state .==. pure 0 .&&. downV
  readyOut = state .==. pure 0
