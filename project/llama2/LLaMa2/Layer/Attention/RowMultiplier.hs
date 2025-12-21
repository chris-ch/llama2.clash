module LLaMa2.Layer.Attention.RowMultiplier
  ( rowMultiplier
  , RowMultiplierOut(..)
  , RowMultiplierDebug(..)
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (ModelDimension, HeadDimension)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (RowI8E)
import qualified Prelude as P
import qualified LLaMa2.Numeric.Operations as OPS
import Clash.Debug (trace)

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
