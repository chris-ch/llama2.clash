module LLaMa2.Layer.Attention.RowComputeUnit
  ( RowComputeIn(..), RowComputeOut(..)
  , rowComputeUnit
  ) where

import Clash.Prelude
import qualified LLaMa2.Layer.Attention.RowMultiplier as RM
import LLaMa2.Numeric.Types
import LLaMa2.Numeric.Quantization
import LLaMa2.Types.ModelConfig
import qualified LLaMa2.Numeric.Operations as OPS
import qualified Prelude as P
import Clash.Debug (trace)

-- | Trace row computation results
traceRowDone :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom (Index HeadDimension) -> Signal dom FixedPoint
  -> Signal dom Bool
traceRowDone layerIdx headIdx rowDone ri result = traced
  where
    traced = go <$> rowDone <*> ri <*> result
    go rd ridx res
      | rd        = trace (prefix P.++ "row=" P.++ show ridx P.++ " result=" P.++ show res) rd
      | otherwise = rd
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

--------------------------------------------------------------------------------
-- BLOCK: RowMultiplier
-- Bundles FSM controller with parallel64RowProcessor
--
-- Inputs:
--   column      - input vector to multiply
--   row         - current row weights
--   colValid    - start signal (latched externally)
--   rowValid    - weights ready signal
--   downReady   - downstream ready
--   rowIndex    - current row (0..HeadDimension-1)
--
-- Outputs:
--   result, rowDone, state, fetchReq, allDone, idleReady, debug signals
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- COMPONENT: RowComputeUnit
-- Executes row-vector multiplication using rowMultiplier
--------------------------------------------------------------------------------
data RowComputeIn dom = RowComputeIn
  { rcInputValid      :: Signal dom Bool
  , rcWeightValid     :: Signal dom Bool
  , rcDownStreamReady :: Signal dom Bool
  , rcRowIndex        :: Signal dom (Index HeadDimension)
  , rcWeight          :: Signal dom (RowI8E ModelDimension)
  , rcColumn          :: Signal dom (Vec ModelDimension FixedPoint)
  } deriving (Generic)

data RowComputeOut dom = RowComputeOut
  { rcResult       :: Signal dom FixedPoint
  , rcResultHC     :: Signal dom FixedPoint  -- HC reference result
  , rcRowDone      :: Signal dom Bool
  , rcAllDone      :: Signal dom Bool
  , rcIdleReady    :: Signal dom Bool
  , rcFetchReq     :: Signal dom Bool
  , rcMultState    :: Signal dom OPS.MultiplierState
  , rcDebug        :: RM.RowMultiplierDebug dom
  } deriving (Generic)

rowComputeUnit :: forall dom.
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Index NumQueryHeads
  -> RowComputeIn dom
  -> RowComputeOut dom
rowComputeUnit layerIdx headIdx inputs =
  RowComputeOut
    { rcResult       = rowDoneTraced `seq` RM.rmoResult mult
    , rcResultHC     = hcRowResult
    , rcRowDone      = RM.rmoRowDone mult
    , rcAllDone      = RM.rmoAllDone mult
    , rcIdleReady    = RM.rmoIdleReady mult
    , rcFetchReq     = RM.rmoFetchReq mult
    , rcMultState    = RM.rmoState mult
    , rcDebug        = RM.rmoDebug mult
    }
  where
    -- Main multiplier for DRAM weights
    mult = RM.rowMultiplier (rcColumn inputs) (rcWeight inputs) 
                         (rcInputValid inputs) (rcWeightValid inputs) 
                         (rcDownStreamReady inputs) (rcRowIndex inputs)

    -- HC reference path (for validation)
    (hcRowResult, _, _) =
      OPS.parallel64RowProcessor
        (RM.rmdRowReset (RM.rmoDebug mult))
        (RM.rmdRowEnable (RM.rmoDebug mult))
        (rcWeight inputs)  -- Use HC weights when available
        (rcColumn inputs)

    -- Row done tracing
    rowDoneTraced = traceRowDone layerIdx headIdx 
                      (RM.rmoRowDone mult) (rcRowIndex inputs) (RM.rmoResult mult)
