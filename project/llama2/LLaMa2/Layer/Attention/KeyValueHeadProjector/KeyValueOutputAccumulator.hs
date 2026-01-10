module LLaMa2.Layer.Attention.KeyValueHeadProjector.KeyValueOutputAccumulator
  ( kvOutputAccumulator
  , KVOutputAccumIn(..)
  , KVOutputAccumOut(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types
import LLaMa2.Types.ModelConfig
import qualified Prelude as P

import TraceUtils (traceWhenC)

--------------------------------------------------------------------------------
-- KVOutputAccumulator
-- Accumulates K and V row results into separate output vectors
--------------------------------------------------------------------------------

data KVOutputAccumIn dom = KVOutputAccumIn
  { kvoaRowDone     :: Signal dom Bool
  , kvoaRowIndex    :: Signal dom (Index HeadDimension)
  , kvoaKResult     :: Signal dom FixedPoint    -- DRAM K row result
  , kvoaVResult     :: Signal dom FixedPoint    -- DRAM V row result
  , kvoaKResultHC   :: Signal dom FixedPoint    -- HC K row result
  , kvoaVResultHC   :: Signal dom FixedPoint    -- HC V row result
  } deriving (Generic)

data KVOutputAccumOut dom = KVOutputAccumOut
  { kvoaKOutput   :: Signal dom (Vec HeadDimension FixedPoint)  -- DRAM K output vector
  , kvoaVOutput   :: Signal dom (Vec HeadDimension FixedPoint)  -- DRAM V output vector
  , kvoaKOutputHC :: Signal dom (Vec HeadDimension FixedPoint)  -- HC K output (validation)
  , kvoaVOutputHC :: Signal dom (Vec HeadDimension FixedPoint)  -- HC V output (validation)
  } deriving (Generic)

-- | Accumulate K and V row results into output vectors
--
-- == Operation
--
-- On each rowDone pulse, stores the K and V results at the current row index
-- in their respective output vectors.
--
-- == Parallel Accumulation
--
-- @
--   rowDone ──────────────────┬────────────────┐
--   rowIndex ────────────────►│                │
--                             ▼                ▼
--   kResult ───────────► kOut[rowIndex]   vOut[rowIndex] ◄─── vResult
-- @
--
kvOutputAccumulator :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumKeyValueHeads      -- Note: KV head index, not Q head
  -> KVOutputAccumIn dom
  -> KVOutputAccumOut dom
kvOutputAccumulator cycleCounter layerIdx kvHeadIdx inputs =
  KVOutputAccumOut
    { kvoaKOutput   = kOut
    , kvoaVOutput   = vOut
    , kvoaKOutputHC = kOutHC
    , kvoaVOutputHC = vOutHC
    }
  where
    tag = "[KVOA L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

    ----------------------------------------------------------------------------
    -- K Output Accumulator (DRAM)
    ----------------------------------------------------------------------------
    
    kOut = register (repeat 0) nextKOutput

    -- Trace K result when rowDone fires
    kResultTraced = traceWhenC cycleCounter (tag P.++ "kResult") 
                               (kvoaRowDone inputs) (kvoaKResult inputs)

    nextKOutput = mux (kvoaRowDone inputs)
                      (replace <$> kvoaRowIndex inputs <*> kResultTraced <*> kOut)
                      kOut

    ----------------------------------------------------------------------------
    -- V Output Accumulator (DRAM)
    ----------------------------------------------------------------------------
    
    vOut = register (repeat 0) nextVOutput

    -- Trace V result when rowDone fires
    vResultTraced = traceWhenC cycleCounter (tag P.++ "vResult") 
                               (kvoaRowDone inputs) (kvoaVResult inputs)

    nextVOutput = mux (kvoaRowDone inputs)
                      (replace <$> kvoaRowIndex inputs <*> vResultTraced <*> vOut)
                      vOut

    ----------------------------------------------------------------------------
    -- K Output Accumulator (HC - validation)
    ----------------------------------------------------------------------------
    
    kOutHC = register (repeat 0) nextKOutputHC

    nextKOutputHC = mux (kvoaRowDone inputs)
                        (replace <$> kvoaRowIndex inputs <*> kvoaKResultHC inputs <*> kOutHC)
                        kOutHC

    ----------------------------------------------------------------------------
    -- V Output Accumulator (HC - validation)
    ----------------------------------------------------------------------------
    
    vOutHC = register (repeat 0) nextVOutputHC

    nextVOutputHC = mux (kvoaRowDone inputs)
                        (replace <$> kvoaRowIndex inputs <*> kvoaVResultHC inputs <*> vOutHC)
                        vOutHC
