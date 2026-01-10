module LLaMa2.Layer.Attention.KeyValueHeadProjector.KVRowComputeUnit
  ( KVRowComputeIn(..)
  , KVRowComputeOut(..)
  , kvRowComputeUnit
  , KVRowMultiplierDebug(..)
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Numeric.Operations as OPS

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- KVRowMultiplier Debug Info
--------------------------------------------------------------------------------

data KVRowMultiplierDebug dom = KVRowMultiplierDebug
  { kvmdAccValueK  :: Signal dom FixedPoint
  , kvmdAccValueV  :: Signal dom FixedPoint
  , kvmdRowReset   :: Signal dom Bool
  , kvmdRowEnable  :: Signal dom Bool
  } deriving (Generic)

--------------------------------------------------------------------------------
-- KVRowComputeUnit Input/Output
--------------------------------------------------------------------------------

data KVRowComputeIn dom = KVRowComputeIn
  { kvrcInputValid      :: Signal dom Bool
  , kvrcWeightValid     :: Signal dom Bool                       -- Both K and V weights ready
  , kvrcDownStreamReady :: Signal dom Bool
  , kvrcRowIndex        :: Signal dom (Index HeadDimension)
  , kvrcKWeightDram     :: Signal dom (RowI8E ModelDimension)    -- DRAM K weights
  , kvrcVWeightDram     :: Signal dom (RowI8E ModelDimension)    -- DRAM V weights
  , kvrcKWeightHC       :: Signal dom (RowI8E ModelDimension)    -- HC K weights (validation)
  , kvrcVWeightHC       :: Signal dom (RowI8E ModelDimension)    -- HC V weights (validation)
  , kvrcColumn          :: Signal dom (Vec ModelDimension FixedPoint)  -- Input vector (xHat)
  } deriving (Generic)

data KVRowComputeOut dom = KVRowComputeOut
  { kvrcKResult     :: Signal dom FixedPoint   -- DRAM K result
  , kvrcVResult     :: Signal dom FixedPoint   -- DRAM V result
  , kvrcKResultHC   :: Signal dom FixedPoint   -- HC K result (validation)
  , kvrcVResultHC   :: Signal dom FixedPoint   -- HC V result (validation)
  , kvrcRowDone     :: Signal dom Bool         -- Row computation complete
  , kvrcAllDone     :: Signal dom Bool         -- All rows complete
  , kvrcIdleReady   :: Signal dom Bool         -- Ready for new request
  , kvrcFetchReq    :: Signal dom Bool         -- Request next weights
  , kvrcMultState   :: Signal dom OPS.MultiplierState
  , kvrcDebug       :: KVRowMultiplierDebug dom
  } deriving (Generic)

--------------------------------------------------------------------------------
-- KVRowComputeUnit
-- Executes K and V row-vector multiplications in parallel
-- Uses shared FSM control but parallel datapaths
--------------------------------------------------------------------------------

-- | Compute unit for K and V projections
--
-- == Architecture
--
-- @
--                      ┌─────────────────────────────────────────┐
--                      │            KVRowComputeUnit             │
--                      │                                         │
--   inputValid ───────►│  ┌───────────────────────────────────┐  │
--   weightValid ──────►│  │         Shared FSM Control        │  │
--   downStreamReady ──►│  │  (matrixMultiplierStateMachine)   │  │
--                      │  └───────────────────────────────────┘  │
--                      │           │ reset  │ enable             │
--                      │           ▼        ▼                    │
--   column (xHat) ────►│  ┌─────────────┐  ┌─────────────┐       │
--   kWeightDram ──────►│  │  K Datapath │  │  V Datapath │       │
--   vWeightDram ──────►│  │  (DRAM+HC)  │  │  (DRAM+HC)  │       │
--                      │  └──────┬──────┘  └──────┬──────┘       │
--                      │         │                │              │
--   kResult ◄──────────│─────────┘                │              │
--   vResult ◄──────────│──────────────────────────┘              │
--   rowDone ◄──────────│                                         │
--                      └─────────────────────────────────────────┘
-- @
--
-- == Key Design Decision
--
-- K and V share the same FSM but have independent datapaths because:
-- 1. Both use the same input column (xHat)
-- 2. Both process the same row index
-- 3. Weights arrive together (from KVWeightFetchUnit)
-- 4. Results are needed together (for attention)
--
-- The FSM is driven by the K datapath's rowDone signal (both complete
-- simultaneously since they process the same number of elements).
--
kvRowComputeUnit :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> KVRowComputeIn dom
  -> KVRowComputeOut dom
kvRowComputeUnit cycleCounter inputs =
  KVRowComputeOut
    { kvrcKResult     = kResult
    , kvrcVResult     = vResult
    , kvrcKResultHC   = kResultHC
    , kvrcVResultHC   = vResultHC
    , kvrcRowDone     = rowDone
    , kvrcAllDone     = allDone
    , kvrcIdleReady   = idleReady
    , kvrcFetchReq    = fetchReq
    , kvrcMultState   = state
    , kvrcDebug       = KVRowMultiplierDebug kAccValue vAccValue rowReset rowEnable
    }
  where
    -- Trace weightValid edges
    weightValidTraced = traceEdgeC cycleCounter "[KVRCU] weightValid" (kvrcWeightValid inputs)

    ----------------------------------------------------------------------------
    -- Shared FSM Control
    -- Uses K's rowDone for state transitions (V completes at same time)
    ----------------------------------------------------------------------------
    
    (state, fetchReq, rowReset, rowEnable, allDone, idleReady) =
      OPS.matrixMultiplierStateMachine 
        (kvrcInputValid inputs) 
        weightValidTraced 
        (kvrcDownStreamReady inputs) 
        kRowDone  -- Use K's done signal (they're synchronized)
        (kvrcRowIndex inputs)

    ----------------------------------------------------------------------------
    -- K Datapath (DRAM - primary computation)
    ----------------------------------------------------------------------------
    
    (kResult, kRowDone, kAccValue) =
      OPS.parallel64RowProcessor rowReset rowEnable 
                                  (kvrcKWeightDram inputs) 
                                  (kvrcColumn inputs)

    -- K Datapath (HC - validation)
    (kResultHC, _, _) =
      OPS.parallel64RowProcessor rowReset rowEnable 
                                  (kvrcKWeightHC inputs) 
                                  (kvrcColumn inputs)

    ----------------------------------------------------------------------------
    -- V Datapath (DRAM - primary computation)
    ----------------------------------------------------------------------------
    
    (vResult, _vRowDone, vAccValue) =
      OPS.parallel64RowProcessor rowReset rowEnable 
                                  (kvrcVWeightDram inputs) 
                                  (kvrcColumn inputs)

    -- V Datapath (HC - validation)
    (vResultHC, _, _) =
      OPS.parallel64RowProcessor rowReset rowEnable 
                                  (kvrcVWeightHC inputs) 
                                  (kvrcColumn inputs)

    -- Use K's rowDone for the output (V is identical timing)
    rowDone = kRowDone