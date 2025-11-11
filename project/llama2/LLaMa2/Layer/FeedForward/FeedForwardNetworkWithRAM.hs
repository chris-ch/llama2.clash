module LLaMa2.Layer.FeedForward.FeedForwardNetworkWithRAM
  ( feedForwardStageWithRAM ) where

import Clash.Prelude
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Types.ModelConfig (ModelDimension, HiddenDimension)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E, dequantRowToF)
import qualified Simulation.Parameters as PARAM (FeedForwardNetworkComponentQ(..))
import LLaMa2.Layer.FeedForward.Activation (sigmoidLinearUnit)
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplierDyn)

feedForwardStageWithRAM ::
  HiddenClockResetEnable dom =>
  Signal dom Bool                              ->  -- validIn
  Signal dom Bool                              ->  -- readyIn (downstream)
  PARAM.FeedForwardNetworkComponentQ                 ->  -- constants fallback
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
  letW1 = mux useRAMFFN ramW1 (pure (PARAM.fW1Q ffn))
  letW2 = mux useRAMFFN ramW2 (pure (PARAM.fW2Q ffn))
  letW3 = mux useRAMFFN ramW3 (pure (PARAM.fW3Q ffn))
  
  -- Normalize with RAM rms when enabled (using dequantRowToF)
  xHat = rmsNormFwFix 
    <$> xIn
    <*> mux useRAMFFN (dequantRowToF <$> ramRms) (pure (PARAM.fRMSFfnF ffn))
  
  -- Rest of the code remains unchanged...
  state = register (0::Unsigned 3) (mux accept (pure 1) (mux (state .==. pure 1 .&&. gateV) (pure 2)
                         (mux (state .==. pure 2 .&&. upV) (pure 3)
                         (mux (state .==. pure 3 .&&. downV) (pure 0) state))))
  accept = (state .==. pure 0) .&&. validIn
  gateValidIn = state .==. pure 1
  upValidIn   = state .==. pure 2
  downValidIn = state .==. pure 3
  (gateRaw, gateV, gr) = parallelRowMatrixMultiplierDyn gateValidIn (pure True) letW1 xHat
  (upRaw,   upV,   ur) = parallelRowMatrixMultiplierDyn upValidIn   (pure True) letW3 xHat
  gateSiLU = regEn (repeat 0) gateV (map sigmoidLinearUnit <$> gateRaw)
  gateUp   = regEn (repeat 0) upV   (zipWith (*) <$> gateSiLU <*> upRaw)
  (downRaw, downV, dr) = parallelRowMatrixMultiplierDyn downValidIn (pure True) letW2 gateUp
  outVec   = regEn (repeat 0) downV downRaw
  validOut = state .==. pure 0 .&&. downV
  readyOut = state .==. pure 0
