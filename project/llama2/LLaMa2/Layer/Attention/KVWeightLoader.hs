module LLaMa2.Layer.Attention.KVWeightLoader
  ( kvWeightLoader
  , KVWeightLoaderOutput(..)
  , KVLoadState(..)
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumKeyValueHeads, NumLayers )
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout
import LLaMa2.Memory.WeightsLayout (FetcherDebug(..))
import qualified Simulation.Parameters as PARAM
import qualified Prelude as P
import Clash.Debug (trace)

--------------------------------------------------------------------------------
-- KV Weight Loader FSM States
--------------------------------------------------------------------------------

-- | FSM states for sequential K/V loading
--
-- @
--   KVIdle ──(request)──► KVFetchingK ──(K done)──► KVFetchingV ──(V done)──► KVDone
--     ▲                                                                         │
--     └─────────────────────(consume)───────────────────────────────────────────┘
-- @
data KVLoadState = KVIdle | KVFetchingK | KVFetchingV | KVDone
  deriving (Show, Eq, Generic, NFDataX)

--------------------------------------------------------------------------------
-- Output Record
--------------------------------------------------------------------------------

data KVWeightLoaderOutput dom = KVWeightLoaderOutput
  { kvHcKRowOut   :: Signal dom (RowI8E ModelDimension)  -- ^ HC K row (for verification)
  , kvHcVRowOut   :: Signal dom (RowI8E ModelDimension)  -- ^ HC V row (for verification)
  , kvDramKRowOut :: Signal dom (RowI8E ModelDimension)  -- ^ DRAM K row
  , kvDramVRowOut :: Signal dom (RowI8E ModelDimension)  -- ^ DRAM V row
  , kvDbgState    :: Signal dom KVLoadState              -- ^ Current FSM state
  , kvDbgKAddr    :: Signal dom (Unsigned 32)            -- ^ K address (debug)
  , kvDbgVAddr    :: Signal dom (Unsigned 32)            -- ^ V address (debug)
  }

--------------------------------------------------------------------------------
-- Transaction Record
--------------------------------------------------------------------------------

data KVTxn = KVTxn
  { kvtRow   :: Index HeadDimension
  , kvtKAddr :: Unsigned 32
  , kvtVAddr :: Unsigned 32
  } deriving (Generic, NFDataX, Show, Eq)

--------------------------------------------------------------------------------
-- KV Weight Loader
--------------------------------------------------------------------------------

-- | Weight loader for K and V matrices with DRAM fetching and HC reference.
--
-- == Overview
--
-- Fetches both K and V rows sequentially using a single AXI master interface.
-- This is more efficient than two separate loaders when K and V are always
-- needed together (which is the case in attention).
--
-- == Timing
--
-- @
-- Cycle:     0    1    2    3    4    5    6    7    8    9   10
-- request:   ─┐   │    │    │    │    │    │    │    │    │    │
--            └───┘    │    │    │    │    │    │    │    │    │
-- state:    Idle  FetchK ─────────── FetchV ─────────── Done  Idle
-- K AXI:          AR   R    R    │    │    │    │    │    │    │
-- V AXI:          │    │    │    AR   R    R    │    │    │    │
-- valid:          │    │    │    │    │    │    ────────┘    │
-- @
--
-- == Interface
--
-- Uses the same handshake protocol as the Q WeightLoader:
-- - Assert rowReqValid when weightReady is True
-- - Wait for weightValid
-- - Assert dataConsumed to return to Idle
--
kvWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)       -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom           -- ^ AXI slave interface (from arbiter)
  -> Index NumLayers                -- ^ Layer index (static)
  -> Index NumKeyValueHeads         -- ^ KV head index (static) - NOTE: Not NumQueryHeads!
  -> Signal dom (Index HeadDimension)  -- ^ Row request
  -> Signal dom Bool                -- ^ rowReqValid (pulse)
  -> Signal dom Bool                -- ^ downstreamReady (level)
  -> Signal dom Bool                -- ^ dataConsumed (pulse)
  -> PARAM.DecoderParameters        -- ^ Model parameters (for HC weights)
  -> ( Master.AxiMasterOut dom
     , KVWeightLoaderOutput dom
     , Signal dom Bool              -- ^ weightValid (level) - both K and V ready
     , Signal dom Bool              -- ^ weightReady (level) - can accept new request
     )
kvWeightLoader cycleCounter dram layerIdx kvHeadIdx rowReq rowReqValid downstreamReady dataConsumed params =
  (axiMaster, out, weightValid, weightReady)
 where
  tag = "[KVWL L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

  ----------------------------------------------------------------------------
  -- Hardcoded (HC) weights for verification
  ----------------------------------------------------------------------------
  
  -- Get the KV head parameters for this layer/head
  layerParams = PARAM.modelLayers params !! layerIdx
  mhaParams = PARAM.multiHeadAttention layerParams
  kvHeadParams = PARAM.kvHeads mhaParams !! kvHeadIdx
  
  hcKWeights :: MatI8E HeadDimension ModelDimension
  hcKWeights = PARAM.kMatrix kvHeadParams
  
  hcVWeights :: MatI8E HeadDimension ModelDimension
  hcVWeights = PARAM.vMatrix kvHeadParams

  ----------------------------------------------------------------------------
  -- FSM State
  ----------------------------------------------------------------------------
  
  loadState :: Signal dom KVLoadState
  loadState = register KVIdle nextState

  weightReady :: Signal dom Bool
  weightReady = loadState .==. pure KVIdle

  weightValid :: Signal dom Bool
  weightValid = loadState .==. pure KVDone

  ----------------------------------------------------------------------------
  -- Address Calculation
  ----------------------------------------------------------------------------
  
  -- Live addresses (combinational, from current row request)
  liveKAddr :: Signal dom (Unsigned 32)
  liveKAddr = Layout.kvRowAddressCalculator Layout.KMatrix layerIdx kvHeadIdx <$> rowReq

  liveVAddr :: Signal dom (Unsigned 32)
  liveVAddr = Layout.kvRowAddressCalculator Layout.VMatrix layerIdx kvHeadIdx <$> rowReq

  ----------------------------------------------------------------------------
  -- Request Handshake
  ----------------------------------------------------------------------------
  
  -- Start fetch when idle, fetcher ready, and request valid
  actualFetchStart :: Signal dom Bool
  actualFetchStart = weightReady .&&. fetcherReady .&&. rowReqValid

  -- Capture transaction on handshake
  txnReg :: Signal dom KVTxn
  txnReg = regEn (KVTxn 0 0 0) actualFetchStart 
                 (KVTxn <$> rowReq <*> liveKAddr <*> liveVAddr)

  ----------------------------------------------------------------------------
  -- AXI Multi-Word Fetcher (shared for K and V)
  ----------------------------------------------------------------------------
  
  -- Detect when K fetch completes (triggers V fetch)
  kFetchComplete :: Signal dom Bool
  kFetchComplete = (loadState .==. pure KVFetchingK) .&&. fetchValid

  -- Address mux: 
  -- - During initial request (Idle): use live K address
  -- - When K fetch completes (kFetchComplete): IMMEDIATELY switch to V address
  --   (this is critical - fetchPulse fires in same cycle as kFetchComplete,
  --    but state is still FetchingK, so we need to detect the transition)
  -- - During V fetch: use captured V address
  currentFetchAddr :: Signal dom (Unsigned 32)
  currentFetchAddr = mux (loadState .==. pure KVIdle) 
                         liveKAddr              -- Use live K addr for initial fetch
                       $ mux kFetchComplete
                             (kvtVAddr <$> txnReg)  -- V addr when transitioning to V fetch
                       $ mux (loadState .==. pure KVFetchingK)
                             (kvtKAddr <$> txnReg)  -- Captured K addr while K is fetching
                             (kvtVAddr <$> txnReg)  -- V addr during V fetch

  -- Fetch pulse: on initial request OR when K fetch completes
  fetchPulse :: Signal dom Bool
  fetchPulse = actualFetchStart .||. kFetchComplete

  (axiMaster, fetchedWords, fetchValidRaw, fetcherReady, fetcherDebug) =
    Layout.axiMultiWordRowFetcher @_ @ModelDimension dram fetchPulse currentFetchAddr

  -- DEBUG: Trace fetcher internal state (force evaluation via seq)
  fetcherDebugTraced = traceFetcherDebug cycleCounter tag fetcherDebug fetchValidRaw
  
  -- Force debug evaluation by making fetchValid depend on it
  fetchValid = forceEval <$> fetcherDebugTraced <*> fetchValidRaw
   where
    forceEval () fv = fv

  ----------------------------------------------------------------------------
  -- Row Parsing and Staging
  ----------------------------------------------------------------------------
  
  zeroRow :: RowI8E ModelDimension
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  -- Parse fetched words
  parsedRow :: Signal dom (RowI8E ModelDimension)
  parsedRow = Layout.multiWordRowParser <$> fetchedWords

  -- K row: capture when K fetch completes
  kFetchDone :: Signal dom Bool
  kFetchDone = (loadState .==. pure KVFetchingK) .&&. fetchValid

  dramKRowAssembled :: Signal dom (RowI8E ModelDimension)
  dramKRowAssembled = regEn zeroRow kFetchDone parsedRow

  -- V row: capture when V fetch completes  
  vFetchDone :: Signal dom Bool
  vFetchDone = (loadState .==. pure KVFetchingV) .&&. fetchValid

  dramVRowAssembled :: Signal dom (RowI8E ModelDimension)
  dramVRowAssembled = regEn zeroRow vFetchDone parsedRow

  -- Commit rows on rising edge of weightValid (transition to Done)
  prevValid = register False weightValid
  dvRise = weightValid .&&. (not <$> prevValid)

  dramKRowCommitted :: Signal dom (RowI8E ModelDimension)
  dramKRowCommitted = regEn zeroRow dvRise dramKRowAssembled

  dramVRowCommitted :: Signal dom (RowI8E ModelDimension)
  dramVRowCommitted = regEn zeroRow dvRise dramVRowAssembled

  -- HC rows: commit at same time as DRAM rows
  hcKRowCommitted :: Signal dom (RowI8E ModelDimension)
  hcKRowCommitted = regEn zeroRow dvRise ((!!) hcKWeights . kvtRow <$> txnReg)

  hcVRowCommitted :: Signal dom (RowI8E ModelDimension)
  hcVRowCommitted = regEn zeroRow dvRise ((!!) hcVWeights . kvtRow <$> txnReg)

  ----------------------------------------------------------------------------
  -- Verification: Compare DRAM vs HC
  ----------------------------------------------------------------------------
  
  -- Delayed check trigger (one cycle after dvRise for stable data)
  dvRiseD1 :: Signal dom Bool
  dvRiseD1 = register False dvRise

  -- K row verification
  dramKRowChecked :: Signal dom (RowI8E ModelDimension)
  dramKRowChecked = assertKVRowMatch cycleCounter dvRiseD1 
                                      (kvtRow <$> txnReg) 
                                      "K" layerIdx kvHeadIdx
                                      dramKRowCommitted hcKRowCommitted

  -- V row verification
  dramVRowChecked :: Signal dom (RowI8E ModelDimension)
  dramVRowChecked = assertKVRowMatch cycleCounter dvRiseD1 
                                      (kvtRow <$> txnReg) 
                                      "V" layerIdx kvHeadIdx
                                      dramVRowCommitted hcVRowCommitted

  ----------------------------------------------------------------------------
  -- FSM Next State Logic
  ----------------------------------------------------------------------------
  
  nextState =
    -- Idle → FetchingK: on new request
    mux (loadState .==. pure KVIdle .&&. actualFetchStart)
        (pure KVFetchingK)
    -- FetchingK → FetchingV: when K fetch completes
    $ mux (loadState .==. pure KVFetchingK .&&. fetchValid)
        (pure KVFetchingV)
    -- FetchingV → Done: when V fetch completes
    $ mux (loadState .==. pure KVFetchingV .&&. fetchValid)
        (pure KVDone)
    -- Done → Idle: when downstream consumes
    $ mux (loadState .==. pure KVDone .&&. downstreamReady .&&. dataConsumed)
        (pure KVIdle)
        loadState

  ----------------------------------------------------------------------------
  -- Tracing
  ----------------------------------------------------------------------------
  
  loadStateTraced = traceKVLoadState cycleCounter tag loadState

  ----------------------------------------------------------------------------
  -- Outputs
  ----------------------------------------------------------------------------
  
  out = KVWeightLoaderOutput
    { kvHcKRowOut   = hcKRowCommitted
    , kvHcVRowOut   = hcVRowCommitted
    , kvDramKRowOut = dramKRowChecked
    , kvDramVRowOut = dramVRowChecked
    , kvDbgState    = loadStateTraced
    , kvDbgKAddr    = kvtKAddr <$> txnReg
    , kvDbgVAddr    = kvtVAddr <$> txnReg
    }

--------------------------------------------------------------------------------
-- Verification Helpers
--------------------------------------------------------------------------------

-- | Assert that DRAM and HC rows match on commit
assertKVRowMatch
  :: Signal dom (Unsigned 32)          -- ^ cycleCounter
  -> Signal dom Bool                   -- ^ check trigger
  -> Signal dom (Index HeadDimension)  -- ^ row index
  -> String                            -- ^ "K" or "V"
  -> Index NumLayers                   -- ^ layer index
  -> Index NumKeyValueHeads            -- ^ KV head index
  -> Signal dom (RowI8E ModelDimension)  -- ^ DRAM row
  -> Signal dom (RowI8E ModelDimension)  -- ^ HC row
  -> Signal dom (RowI8E ModelDimension)  -- ^ Checked DRAM row (or error)
assertKVRowMatch cyc trigger rowIdx matName layerIdx' kvHeadIdx' dramRow hcRow =
  mux trigger (check <$> cyc <*> rowIdx <*> dramRow <*> hcRow) dramRow
 where
  check c ri dr hr
    | rowExponent dr P.== rowExponent hr P.&& rowMantissas dr P.== rowMantissas hr = dr
    | otherwise = P.error $
        "@" P.++ show c P.++ " " P.++ matName P.++ " DRAM/HC mismatch!"
        P.++ "\n  layer=" P.++ show layerIdx'
        P.++ " kvHead=" P.++ show kvHeadIdx'
        P.++ " row=" P.++ show ri
        P.++ "\n  DRAM exp=" P.++ show (rowExponent dr)
        P.++ " HC exp=" P.++ show (rowExponent hr)
        P.++ "\n  DRAM mant[0..7]=" P.++ show (P.take 8 $ P.map show $ toList $ rowMantissas dr)
        P.++ "\n  HC mant[0..7]=" P.++ show (P.take 8 $ P.map show $ toList $ rowMantissas hr)

-- | Trace FSM state changes
traceKVLoadState
  :: forall dom . HiddenClockResetEnable dom => Signal dom (Unsigned 32)
  -> String
  -> Signal dom KVLoadState
  -> Signal dom KVLoadState
traceKVLoadState cyc tag' sig = result
 where
  prev = register KVIdle sig
  result = emit <$> cyc <*> sig <*> prev
  emit c curr pr
    | curr P./= pr = trace ("@" P.++ show c P.++ " " P.++ tag' P.++ "state: " 
                           P.++ show pr P.++ " → " P.++ show curr) curr
    | otherwise = curr

-- | Trace fetcher debug signals for diagnosing state machine issues
traceFetcherDebug
  :: forall dom . HiddenClockResetEnable dom => Signal dom (Unsigned 32)
  -> String
  -> FetcherDebug dom
  -> Signal dom Bool  -- fetchValid
  -> Signal dom ()
traceFetcherDebug cyc tag' dbg fetchValid = result
 where
  -- Track previous values for edge detection
  prevRReceived = register False (dbgRReceived dbg)
  prevDoneCondition = register False (dbgDoneCondition dbg)
  prevFetchValid = register False fetchValid
  
  result = emit <$> cyc 
                <*> dbgRReceived dbg <*> prevRReceived
                <*> dbgBeat dbg
                <*> dbgStateIsWaitR dbg
                <*> dbgDoneCondition dbg <*> prevDoneCondition
                <*> fetchValid <*> prevFetchValid
  
  emit c rRcv prevRRcv beat isWaitR doneCond prevDoneCond fv prevFv
    -- Trace when R beat is received
    | rRcv P.&& P.not prevRRcv = 
        trace ("@" P.++ show c P.++ " " P.++ tag' P.++ "FETCHER: R received, beat=" 
               P.++ show beat P.++ " isWaitR=" P.++ show isWaitR
               P.++ " doneCond=" P.++ show doneCond) ()
    -- Trace when done condition becomes true
    | doneCond P.&& P.not prevDoneCond =
        trace ("@" P.++ show c P.++ " " P.++ tag' P.++ "FETCHER: DONE condition TRUE!") ()
    -- Trace when fetchValid rises
    | fv P.&& P.not prevFv =
        trace ("@" P.++ show c P.++ " " P.++ tag' P.++ "FETCHER: fetchValid RISE") ()
    | otherwise = ()
