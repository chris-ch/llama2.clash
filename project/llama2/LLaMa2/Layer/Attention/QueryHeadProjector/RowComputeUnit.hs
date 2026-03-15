module LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit
  ( RowComputeIn(..)
  , RowComputeOut(..)
  , rowComputeUnit
  , RowMultiplierDebug(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (RowI8E)
import qualified LLaMa2.Numeric.Operations as OPS


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

rowMultiplier :: forall dom numRows numCols.
  ( HiddenClockResetEnable dom
  , KnownNat numRows
  , KnownNat numCols
  )
  => Signal dom (Unsigned 32)
  -> Signal dom (Vec numCols FixedPoint)
  -> Signal dom (RowI8E numCols)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index numRows)
  -> RowMultiplierOut dom
rowMultiplier _cycleCounter column row colValid rowValid downReady rowIndex =
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
    (rowResult, rowDone, accValue) =
      OPS.parallel64RowProcessor rowReset rowEnable row column

    (state, fetchReq, rowReset, rowEnable, allDone, idleReady) =
      OPS.matrixMultiplierStateMachine colValid rowValid downReady rowDone rowIndex

--------------------------------------------------------------------------------
-- RowComputeUnit
--------------------------------------------------------------------------------
data RowComputeIn dom numRows numCols = RowComputeIn
  { rcInputValid      :: Signal dom Bool
  , rcWeightValid     :: Signal dom Bool
  , rcDownStreamReady :: Signal dom Bool
  , rcRowIndex        :: Signal dom (Index numRows)
  , rcWeightDram      :: Signal dom (RowI8E numCols)
  , rcColumn          :: Signal dom (Vec numCols FixedPoint)
  } deriving (Generic)

data RowComputeOut dom = RowComputeOut
  { rcResult       :: Signal dom FixedPoint
  , rcRowDone      :: Signal dom Bool
  , rcAllDone      :: Signal dom Bool
  , rcIdleReady    :: Signal dom Bool
  , rcFetchReq     :: Signal dom Bool
  , rcMultState    :: Signal dom OPS.MultiplierState
  , rcDebug        :: RowMultiplierDebug dom
  } deriving (Generic)

{-# NOINLINE rowComputeUnit #-}
rowComputeUnit :: forall dom numRows numCols.
  ( HiddenClockResetEnable dom
  , KnownNat numRows
  , KnownNat numCols
  )
  => Signal dom (Unsigned 32)
  -> RowComputeIn dom numRows numCols
  -> RowComputeOut dom
rowComputeUnit cycleCounter inputs =
  RowComputeOut
    { rcResult       = rmoResult mult
    , rcRowDone      = rmoRowDone mult
    , rcAllDone      = rmoAllDone mult
    , rcIdleReady    = rmoIdleReady mult
    , rcFetchReq     = rmoFetchReq mult
    , rcMultState    = rmoState mult
    , rcDebug        = rmoDebug mult
    }
  where
    mult = rowMultiplier cycleCounter (rcColumn inputs) (rcWeightDram inputs)
                         (rcInputValid inputs) (rcWeightValid inputs)
                         (rcDownStreamReady inputs) (rcRowIndex inputs)
