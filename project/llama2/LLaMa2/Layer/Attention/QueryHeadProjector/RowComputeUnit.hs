module LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit
  ( RowComputeIn(..)
  , RowComputeOut(..)
  , rowComputeUnit
  , RowMultiplierDebug(..)
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Numeric.Operations as OPS

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- RowMultiplier types
--------------------------------------------------------------------------------
data RowMultiplierDebug dom = RowMultiplierDebug
  { rmdAccValue  :: Signal dom FixedPoint
  , rmdRowReset  :: Signal dom Bool
  , rmdRowEnable :: Signal dom Bool
  } deriving (Generic)

data RowMultiplierOut dom = RowMultiplierOut
  { rmoResult     :: Signal dom FixedPoint
  , rmoRowDone    :: Signal dom Bool
  , rmoState      :: Signal dom OPS.MultiplierState
  , rmoFetchReq   :: Signal dom Bool
  , rmoAllDone    :: Signal dom Bool
  , rmoIdleReady  :: Signal dom Bool
  , rmoDebug      :: RowMultiplierDebug dom
  } deriving (Generic)

rowMultiplier :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> RowMultiplierOut dom
rowMultiplier cycleCounter column row colValid rowValid downReady rowIndex =
  RowMultiplierOut
    { rmoResult     = rowResult
    , rmoRowDone    = rowDone
    , rmoState      = state
    , rmoFetchReq   = fetchReq
    , rmoAllDone    = allDone
    , rmoIdleReady  = idleReady
    , rmoDebug      = RowMultiplierDebug accValue rowReset rowEnable
    }
  where
    -- Trace rowValid edges
    rowValidTraced = traceEdgeC cycleCounter "[RCU] rowValid" rowValid

    -- Core computation
    (rowResult, rowDone, accValue) =
      OPS.parallel64RowProcessor rowReset rowEnable row column

    -- FSM control
    (state, fetchReq, rowReset, rowEnable, allDone, idleReady) =
      OPS.matrixMultiplierStateMachine colValid rowValidTraced downReady rowDone rowIndex

--------------------------------------------------------------------------------
-- RowComputeUnit
-- Executes row-vector multiplication with DRAM weights for computation
-- and HC weights for parallel validation
--------------------------------------------------------------------------------
data RowComputeIn dom = RowComputeIn
  { rcInputValid      :: Signal dom Bool
  , rcWeightValid     :: Signal dom Bool
  , rcDownStreamReady :: Signal dom Bool
  , rcRowIndex        :: Signal dom (Index HeadDimension)
  , rcWeightDram      :: Signal dom (RowI8E ModelDimension)  -- DRAM weights for computation
  , rcWeightHC        :: Signal dom (RowI8E ModelDimension)  -- HC weights for validation
  , rcColumn          :: Signal dom (Vec ModelDimension FixedPoint)
  } deriving (Generic)

data RowComputeOut dom = RowComputeOut
  { rcResult       :: Signal dom FixedPoint   -- DRAM-computed result (primary)
  , rcResultHC     :: Signal dom FixedPoint   -- HC-computed result (validation)
  , rcRowDone      :: Signal dom Bool
  , rcAllDone      :: Signal dom Bool
  , rcIdleReady    :: Signal dom Bool
  , rcFetchReq     :: Signal dom Bool
  , rcMultState    :: Signal dom OPS.MultiplierState
  , rcDebug        :: RowMultiplierDebug dom
  } deriving (Generic)

rowComputeUnit :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> RowComputeIn dom
  -> RowComputeOut dom
rowComputeUnit cycleCounter inputs =
  RowComputeOut
    { rcResult       = rmoResult mult    -- DRAM result is primary
    , rcResultHC     = hcRowResult       -- HC result for validation
    , rcRowDone      = rmoRowDone mult
    , rcAllDone      = rmoAllDone mult
    , rcIdleReady    = rmoIdleReady mult
    , rcFetchReq     = rmoFetchReq mult
    , rcMultState    = rmoState mult
    , rcDebug        = rmoDebug mult
    }
  where
    -- Main multiplier using DRAM weights (primary computation path)
    mult = rowMultiplier cycleCounter (rcColumn inputs) (rcWeightDram inputs)
                         (rcInputValid inputs) (rcWeightValid inputs) 
                         (rcDownStreamReady inputs) (rcRowIndex inputs)

    -- HC reference path (for validation)
    (hcRowResult, _, _) =
      OPS.parallel64RowProcessor
        (rmdRowReset (rmoDebug mult))
        (rmdRowEnable (rmoDebug mult))
        (rcWeightHC inputs)
        (rcColumn inputs)
