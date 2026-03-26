module LLaMa2.Layer.Attention.KeyValueHeadProjector.RowComputeUnit
  ( RowComputeIn(..)
  , RowComputeOut(..)
  , rowComputeUnit
  , RowMultiplierDebug(..)
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (HeadDimension, ModelDimension)
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.Quantization (RowI8E (..))
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
  , rmoColumnAddr :: Signal dom (Index ModelDimension)
  } deriving (Generic)

-- | Serial row multiplier: reads one column element per cycle from a BRAM.
--
-- Timing inside one row:
--   MReset   (1 cycle) : column counter reset, addr 0 issued → col[0] arrives in
--                         first MProcessing cycle.
--   MProcessing cycle k: accumulate mant[k] * col[k]; issue addr k+1.
--   rowDone fires 1 cycle after the last element (rising-edge detect + register).
rowMultiplier :: forall dom.
  ( HiddenClockResetEnable dom
  , 1 <= ModelDimension
  )
  => Signal dom (Unsigned 32)
  -> Signal dom FixedPoint             -- ^ colRdData: column BRAM output (1-cycle latency)
  -> Signal dom (RowI8E ModelDimension) -- ^ weight row
  -> Signal dom Bool                   -- ^ colValid
  -> Signal dom Bool                   -- ^ rowValid
  -> Signal dom Bool                   -- ^ downReady
  -> Signal dom (Index HeadDimension)  -- ^ rowIndex
  -> RowMultiplierOut dom
rowMultiplier _cycleCounter colRdData row colValid rowValid downReady rowIndex =
  RowMultiplierOut
    { rmoResult     = rowResult
    , rmoRowDone    = rowDone
    , rmoState      = state
    , rmoFetchReq   = fetchReq
    , rmoAllDone    = allDone
    , rmoIdleReady  = idleReady
    , rmoDebug      = RowMultiplierDebug acc rowReset rowEnable
    , rmoColumnAddr = colAddr
    }
  where
    (state, fetchReq, rowReset, rowEnable, allDone, idleReady) =
      OPS.matrixMultiplierStateMachine colValid rowValid downReady rowDone rowIndex

    -- Column counter: reset on MReset, advance each MProcessing cycle.
    colCounter :: Signal dom (Index ModelDimension)
    colCounter = register 0 nextColCounter

    nextColCounter :: Signal dom (Index ModelDimension)
    nextColCounter =
      mux rowReset (pure 0) $
      mux rowEnable (satSucc SatBound <$> colCounter) $
      colCounter

    -- Pre-fetch address: issued 1 cycle before the element is needed.
    --   MReset  → issue 0  (col[0] arrives in first MProcessing cycle)
    --   MProc k → issue k+1 (col[k+1] arrives next cycle; saturates at maxBound)
    colAddr :: Signal dom (Index ModelDimension)
    colAddr = mux rowReset (pure 0) (satSucc SatBound <$> colCounter)

    -- Mantissa element for current column counter position.
    mantissaElem :: Signal dom (Signed 8)
    mantissaElem = (!!) <$> (rowMantissas <$> row) <*> colCounter

    -- Serial product: mantissa × column element.
    product_ :: Signal dom FixedPoint
    product_ = (fromIntegral <$> mantissaElem) * colRdData

    -- Accumulator: reset on MReset, accumulate during MProcessing (guarded by rowDone).
    acc :: Signal dom FixedPoint
    acc = register 0 nextAcc

    nextAcc :: Signal dom FixedPoint
    nextAcc =
      mux rowReset (pure 0) $
      mux (rowEnable .&&. (not <$> rowDone)) (acc + product_) $
      acc

    -- Row-done pulse: fires 1 cycle after the last element (rising-edge detect).
    lastElemFlag :: Signal dom Bool
    lastElemFlag = (colCounter .==. pure maxBound) .&&. rowEnable

    rowDoneRaw :: Signal dom Bool
    rowDoneRaw = lastElemFlag .&&. (not <$> register False lastElemFlag)

    rowDone :: Signal dom Bool
    rowDone = register False rowDoneRaw

    -- Scaled result (valid when rowDone fires; acc holds the full partial sum).
    rowResult :: Signal dom FixedPoint
    rowResult = scalePow2F <$> (rowExponent <$> row) <*> acc

--------------------------------------------------------------------------------
-- RowComputeUnit
--------------------------------------------------------------------------------
data RowComputeIn dom = RowComputeIn
  { rcInputValid      :: Signal dom Bool
  , rcWeightValid     :: Signal dom Bool
  , rcDownStreamReady :: Signal dom Bool
  , rcRowIndex        :: Signal dom (Index HeadDimension)
  , rcWeightDram      :: Signal dom (RowI8E ModelDimension)
  , rcColumnRdData    :: Signal dom FixedPoint   -- ^ one column element per cycle (BRAM output)
  } deriving (Generic)

data RowComputeOut dom = RowComputeOut
  { rcResult       :: Signal dom FixedPoint
  , rcRowDone      :: Signal dom Bool
  , rcAllDone      :: Signal dom Bool
  , rcIdleReady    :: Signal dom Bool
  , rcFetchReq     :: Signal dom Bool
  , rcMultState    :: Signal dom OPS.MultiplierState
  , rcDebug        :: RowMultiplierDebug dom
  , rcColumnAddr   :: Signal dom (Index ModelDimension)  -- ^ drive to column BRAM read port
  } deriving (Generic)

{-# NOINLINE rowComputeUnit #-}
rowComputeUnit :: forall dom.
  ( HiddenClockResetEnable dom
  , 1 <= ModelDimension
  )
  => Signal dom (Unsigned 32)
  -> RowComputeIn dom
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
    , rcColumnAddr   = rmoColumnAddr mult
    }
  where
    mult = rowMultiplier cycleCounter (rcColumnRdData inputs) (rcWeightDram inputs)
                         (rcInputValid inputs) (rcWeightValid inputs)
                         (rcDownStreamReady inputs) (rcRowIndex inputs)
