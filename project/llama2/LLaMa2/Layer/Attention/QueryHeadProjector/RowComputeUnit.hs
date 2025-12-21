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
import qualified Prelude as P
import Clash.Debug (trace)

--------------------------------------------------------------------------------
-- RowMultiplier types (kept here since RowComputeUnit wraps it)
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
  => Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> RowMultiplierOut dom
rowMultiplier column row colValid rowValid downReady rowIndex =
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
    -- Detect rowValid rising edge for debug
    rowValidRise = rowValid .&&. (not <$> register False rowValid)

    colValidTraced = go <$> rowValidRise <*> colValid
      where
        go True cv = trace ("MULT: rowValid ROSE, colValid=" P.++ show cv) cv
        go False cv = cv

    -- Core computation
    (rowResult, rowDone, accValue) =
      OPS.parallel64RowProcessor rowReset rowEnable row column

    -- FSM control
    (state, fetchReq, rowReset, rowEnable, allDone, idleReady) =
      OPS.matrixMultiplierStateMachine colValidTraced rowValid downReady rowDone rowIndex

--------------------------------------------------------------------------------
-- RowComputeUnit
-- Executes row-vector multiplication with DRAM and HC validation paths
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
  , rcDebug        :: RowMultiplierDebug dom
  } deriving (Generic)

rowComputeUnit :: forall dom.
  HiddenClockResetEnable dom
  => RowComputeIn dom
  -> RowComputeOut dom
rowComputeUnit inputs =
  RowComputeOut
    { rcResult       = rmoResult mult
    , rcResultHC     = hcRowResult
    , rcRowDone      = rmoRowDone mult
    , rcAllDone      = rmoAllDone mult
    , rcIdleReady    = rmoIdleReady mult
    , rcFetchReq     = rmoFetchReq mult
    , rcMultState    = rmoState mult
    , rcDebug        = rmoDebug mult
    }
  where
    -- Main multiplier for DRAM weights
    mult = rowMultiplier (rcColumn inputs) (rcWeight inputs) 
                         (rcInputValid inputs) (rcWeightValid inputs) 
                         (rcDownStreamReady inputs) (rcRowIndex inputs)

    -- HC reference path (for validation)
    (hcRowResult, _, _) =
      OPS.parallel64RowProcessor
        (rmdRowReset (rmoDebug mult))
        (rmdRowEnable (rmoDebug mult))
        (rcWeight inputs)  -- Use same weights (HC for now)
        (rcColumn inputs)
