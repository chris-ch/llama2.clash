module LLaMa2.Layer.Attention.KeyValueHeadProjector.KVWeightFetchUnit
  ( KVWeightFetchIn(..)
  , KVWeightFetchOut(..)
  , kvWeightFetchUnit
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Layer.Attention.KVWeightLoader as KVLOADER
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified Simulation.Parameters as PARAM
import qualified Prelude as P

import TraceUtils (traceEdgeC)
import Debug.Trace (trace)

--------------------------------------------------------------------------------
-- KVWeightFetchUnit
-- Coordinates DRAM weight loading for K and V via KVWeightLoader
--------------------------------------------------------------------------------

data KVWeightFetchIn dom = KVWeightFetchIn
  { kvwfRowIndex      :: Signal dom (Index HeadDimension)  -- ^ Current row to fetch
  , kvwfRowReqValid   :: Signal dom Bool                   -- ^ UNGATED request from compute
  , kvwfConsumeSignal :: Signal dom Bool                   -- ^ Coordinated consume signal
  , kvwfRowDone       :: Signal dom Bool                   -- ^ Row computation complete
  , kvwfInputValid    :: Signal dom Bool                   -- ^ Input valid (for API compat)
  } deriving (Generic)

data KVWeightFetchOut dom = KVWeightFetchOut
  { kvwfAxiMaster   :: Master.AxiMasterOut dom              -- ^ AXI master to arbiter
  , kvwfKWeightDram :: Signal dom (RowI8E ModelDimension)   -- ^ DRAM K row
  , kvwfVWeightDram :: Signal dom (RowI8E ModelDimension)   -- ^ DRAM V row
  , kvwfKWeightHC   :: Signal dom (RowI8E ModelDimension)   -- ^ HC K row (verification)
  , kvwfVWeightHC   :: Signal dom (RowI8E ModelDimension)   -- ^ HC V row (verification)
  , kvwfWeightValid :: Signal dom Bool                      -- ^ Both K and V ready
  , kvwfIdleReady   :: Signal dom Bool                      -- ^ Can accept new request
  } deriving (Generic)

-- | KV Weight Fetch Unit
--
-- Wraps KVWeightLoader with request gating logic, similar to the Q WeightFetchUnit.
--
-- == Request Protocol
--
-- The compute unit asserts kvwfRowReqValid when it needs weights.
-- This unit gates the request to ensure:
-- 1. Only one request is issued per row
-- 2. Requests wait until the loader is idle
-- 3. Edge detection prevents duplicate requests
--
-- == Usage
--
-- @
-- kvWeightFetch = kvWeightFetchUnit cycleCounter dramSlaveIn layerIdx kvHeadIdx params
--                   KVWeightFetchIn
--                     { kvwfRowIndex      = rowIndex
--                     , kvwfRowReqValid   = computeNeedsWeights
--                     , kvwfConsumeSignal = consumeSignal
--                     , kvwfRowDone       = rowDone
--                     , kvwfInputValid    = inputValid
--                     }
--
-- kWeight = kvwfKWeightDram kvWeightFetch
-- vWeight = kvwfVWeightDram kvWeightFetch
-- weightsReady = kvwfWeightValid kvWeightFetch
-- @
--
kvWeightFetchUnit :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)       -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom           -- ^ AXI slave interface
  -> Index NumLayers                -- ^ Layer index (static)
  -> Index NumKeyValueHeads         -- ^ KV head index (static)
  -> PARAM.DecoderParameters        -- ^ Model parameters
  -> KVWeightFetchIn dom            -- ^ Input signals
  -> KVWeightFetchOut dom           -- ^ Output signals
kvWeightFetchUnit cycleCounter dramSlaveIn layerIdx kvHeadIdx params inputs =
  KVWeightFetchOut
    { kvwfAxiMaster   = axiMaster
    , kvwfKWeightDram = kRowDramChecked
    , kvwfVWeightDram = vRowDramChecked
    , kvwfKWeightHC   = kRowHC
    , kvwfVWeightHC   = vRowHC
    , kvwfWeightValid = weightValid
    , kvwfIdleReady   = weightReady
    }
  where
    tag = "[KVWFU L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

    ----------------------------------------------------------------------------
    -- KV Weight Loader
    ----------------------------------------------------------------------------
    
    (axiMaster, loaderOut, weightValidRaw, weightReadyRaw) =
        KVLOADER.kvWeightLoader cycleCounter dramSlaveIn layerIdx kvHeadIdx
                                (kvwfRowIndex inputs)
                                rowReqPulseTraced
                                (pure True)              -- Always ready for next row
                                (kvwfRowDone inputs)
                                params

    -- Trace weight valid and ready edges
    weightValid = traceEdgeC cycleCounter (tag P.++ "weightValid") weightValidRaw
    weightReady = traceEdgeC cycleCounter (tag P.++ "weightReady") weightReadyRaw

    ----------------------------------------------------------------------------
    -- Request Gating Logic
    -- (Same pattern as Q WeightFetchUnit)
    ----------------------------------------------------------------------------
    
    -- Detect when loader transitions back to idle
    loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)

    -- Gate the request internally
    rowReqValidGated = kvwfRowReqValid inputs .&&. weightReady

    -- Track previous state, reset when loader becomes idle
    prevRowReqValid = register False $
        mux loaderBecameIdle (pure False) rowReqValidGated

    rowReqRise = rowReqValidGated .&&. (not <$> prevRowReqValid)

    -- Track row index changes
    prevRowIndex = register 0 $
        mux loaderBecameIdle (kvwfRowIndex inputs) (kvwfRowIndex inputs)

    rowIndexChanged = kvwfRowIndex inputs ./=. prevRowIndex

    -- Generate request pulse on rising edge OR row index change
    rowReqPulse = rowReqRise .||. (rowReqValidGated .&&. rowIndexChanged)

    -- Trace the request pulse
    rowReqPulseTraced = traceEdgeC cycleCounter (tag P.++ "reqPulse") rowReqPulse

    ----------------------------------------------------------------------------
    -- Extract Rows from Loader Output
    ----------------------------------------------------------------------------
    
    kRowDramRaw = KVLOADER.kvDramKRowOut loaderOut
    vRowDramRaw = KVLOADER.kvDramVRowOut loaderOut
    kRowHC      = KVLOADER.kvHcKRowOut loaderOut
    vRowHC      = KVLOADER.kvHcVRowOut loaderOut

    -- Assert rows don't change while valid
    kRowDramStable = assertRowStable weightValid kRowDramRaw
    vRowDramStable = assertRowStable weightValid vRowDramRaw

    -- Weight mismatch checker (K)
    kRowDramChecked = kvWeightMismatchChecker cycleCounter layerIdx kvHeadIdx 
                                               "K" weightValid 
                                               kRowDramStable kRowHC

    -- Weight mismatch checker (V)
    vRowDramChecked = kvWeightMismatchChecker cycleCounter layerIdx kvHeadIdx 
                                               "V" weightValid 
                                               vRowDramStable vRowHC

--------------------------------------------------------------------------------
-- Helper: Assert row stability while valid
--------------------------------------------------------------------------------

assertRowStable :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool
  -> Signal dom (RowI8E n)
  -> Signal dom (RowI8E n)
assertRowStable validSig rowSig = checked
 where
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }
  prevRow = register zeroRow rowSig
  checked = check <$> validSig <*> rowSig <*> prevRow
  check v r pr = if not v || (r == pr) then r
                 else P.error "KV Row changed while valid (loader/consumer desync)"

--------------------------------------------------------------------------------
-- Helper: Weight mismatch checker (assertion)
--------------------------------------------------------------------------------

kvWeightMismatchChecker 
  :: Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> String                              -- ^ "K" or "V"
  -> Signal dom Bool
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
kvWeightMismatchChecker cycleCounter layerIdx' kvHeadIdx' matName valid dram hc = result
  where
    result = check <$> cycleCounter <*> valid <*> dram <*> hc
    check cyc v d h
      | v && (rowExponent d P./= rowExponent h P.|| rowMantissas d P./= rowMantissas h) =
          trace ("@" P.++ show cyc P.++ " [KVWFU L" P.++ show layerIdx' 
                P.++ " KV" P.++ show kvHeadIdx' P.++ "] " P.++ matName 
                P.++ "_WEIGHT_MISMATCH exp_d=" P.++ show (rowExponent d)
                P.++ " exp_h=" P.++ show (rowExponent h)) d
      | otherwise = d
