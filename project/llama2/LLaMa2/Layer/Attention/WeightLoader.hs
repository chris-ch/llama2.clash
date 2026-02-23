module LLaMa2.Layer.Attention.WeightLoader
  ( qWeightLoader      -- Q weights
  , kWeightLoader     -- K weights (for KV heads)
  , vWeightLoader     -- V weights (for KV heads)
  , WeightLoaderOutput(..)
  , LoadState(..)
  , assertRowStable
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumQueryHeads, NumLayers, NumKeyValueHeads )
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified Simulation.Parameters as PARAM
import qualified Prelude as P
import Data.Type.Bool (If)
import Data.Type.Ord (OrdCond)
import qualified GHC.TypeNats as T
import Clash.Debug (trace)

data LoadState = LIdle | LFetching | LDone
  deriving (Show, Eq, Generic, NFDataX)

data WeightLoaderOutput dom = WeightLoaderOutput
  { hcRowOut          :: Signal dom (RowI8E ModelDimension)
  , dramRowOut        :: Signal dom (RowI8E ModelDimension)
  , dbgRequestedAddr  :: Signal dom (Unsigned 32)
  , dbgCapturedAddr   :: Signal dom (Unsigned 32)
  , dbgCapturedRowReq :: Signal dom (Index HeadDimension)
  , dbgLoadState      :: Signal dom LoadState
  }

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
                 else P.error "Row changed while valid (loader/consumer)"

-- | Weight loader for Q matrices (backward compatible API)
qWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)  -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool       -- ^ rowReqValid (pulse)
  -> Signal dom Bool       -- ^ downstreamReady (level)
  -> Signal dom Bool       -- ^ dataConsumed (pulse)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool     -- ^ weightValid (level)
     , Signal dom Bool     -- ^ weightReady (level)
     )
qWeightLoader cycleCounter dram layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed params =
  weightLoaderGeneric cycleCounter dram Layout.QMatrix layerIdx (fromIntegral headIdx) 
                      rowReq rowReqValid downstreamReady dataConsumed hcWeights tagStr
 where
  hcWeights :: MatI8E HeadDimension ModelDimension
  hcWeights =
    PARAM.qMatrix
      (PARAM.qHeads (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)
  tagStr = "[WFU L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

-- | Weight loader for K matrices.
--
-- Delegates to 'weightLoaderGeneric' with K-specific address calculation
-- and HC weights.  K thereby gets the same full verification coverage as Q:
--
--   * ARADDR assertion ('assertArAddrMatchGeneric')
--   * Full commit-time mismatch diagnostics ('assertRowsMatchOnCommitGeneric')
--   * Trace events on fetch start and row commit
--
-- NOTE: This function provides the DRAM-fetching infrastructure for K, but
-- it is not yet wired into 'KeyValueHeadProjector', which still uses
-- hardwired (HC) parameters. Integrating this loader is the next migration
-- step for K.
kWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool
     , Signal dom Bool
     )
kWeightLoader cycleCounter dram layerIdx kvHeadIdx rowReq rowReqValid downstreamReady dataConsumed params =
  weightLoaderGeneric cycleCounter dram Layout.KMatrix layerIdx (fromIntegral kvHeadIdx)
                      rowReq rowReqValid downstreamReady dataConsumed hcWeights tagStr
 where
  hcWeights :: MatI8E HeadDimension ModelDimension
  hcWeights = PARAM.kMatrix
    (PARAM.kvHeads (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! kvHeadIdx)
  tagStr = "[KWFU L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

-- | Weight loader for V matrices.
--
-- Mirrors 'kWeightLoader' exactly, with V-specific address calculation.
-- Delegates to 'weightLoaderGeneric', giving V the same full verification
-- coverage as Q (ARADDR assertion, full commit diagnostics, trace events).
--
-- NOTE: Like 'kWeightLoader', this is not yet wired into
-- 'KeyValueHeadProjector'. Wiring it in is part of the V DRAM migration.
vWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumKeyValueHeads
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool
     , Signal dom Bool
     )
vWeightLoader cycleCounter dram layerIdx kvHeadIdx rowReq rowReqValid downstreamReady dataConsumed params =
  weightLoaderGeneric cycleCounter dram Layout.VMatrix layerIdx (fromIntegral kvHeadIdx)
                      rowReq rowReqValid downstreamReady dataConsumed hcWeights tagStr
 where
  hcWeights :: MatI8E HeadDimension ModelDimension
  hcWeights = PARAM.vMatrix
    (PARAM.kvHeads (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! kvHeadIdx)
  tagStr = "[VWFU L" P.++ show layerIdx P.++ " KV" P.++ show kvHeadIdx P.++ "] "

-- | Generic weight loader that works for any matrix type (Q, K, or V)
--
-- This is the core implementation. The specific loaders (weightLoader, kWeightLoader, 
-- vWeightLoader) are thin wrappers that provide the right HC weights and tags.
--
weightLoaderGeneric :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)  -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom
  -> Layout.MatrixType         -- ^ Which matrix (QMatrix, KMatrix, VMatrix)
  -> Index NumLayers
  -> Int                       -- ^ Head index as Int (works for both Q and KV indices)
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool           -- ^ rowReqValid (pulse)
  -> Signal dom Bool           -- ^ downstreamReady (level)
  -> Signal dom Bool           -- ^ dataConsumed (pulse)
  -> MatI8E HeadDimension ModelDimension  -- ^ HC weights for verification
  -> String                    -- ^ Tag string for tracing
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool         -- ^ weightValid (level)
     , Signal dom Bool         -- ^ weightReady (level)
     )
weightLoaderGeneric cycleCounter dram matrixType layerIdx headIdxInt rowReq rowReqValid downstreamReady dataConsumed hcWeights tagStr =
  (axiMaster, out, weightValid, weightReady)
 where
  -- Loader FSM
  loadState :: Signal dom LoadState
  loadState = register LIdle nextState

  -- Expose loader-level ready (for external interface)
  weightReady :: Signal dom Bool
  weightReady = loadState .==. pure LIdle

  weightValid :: Signal dom Bool
  weightValid = loadState .==. pure LDone

  -- Rising edge when a new row becomes valid to the downstream
  prevValid = register False weightValid
  dvRise    = weightValid .&&. (not <$> prevValid)

  dvRiseD1 :: Signal dom Bool
  dvRiseD1 = register False dvRise  -- one cycle delayed

  -- Live request and address (combinational)
  -- Use the internal address calculator that takes Int for head index
  liveRow  :: Signal dom (Index HeadDimension)
  liveRow  = rowReq

  liveAddr :: Signal dom (Unsigned 32)
  liveAddr = Layout.rowAddressCalculator matrixType layerIdx headIdxInt <$> liveRow

  -- The actual fetch start: requires BOTH loader idle AND fetcher ready
  actualFetchStart :: Signal dom Bool
  actualFetchStart = weightReady .&&. fetcherReady .&&. rowReqValid

  -- AXI multi-word fetcher
  (axiMaster, fetchedWords, fetchValid, fetcherReady, fetcherDebug) =
    Layout.axiMultiWordRowFetcher @_ @ModelDimension dram actualFetchStart liveAddr

  -- Transaction record captured on THE SAME handshake as the fetcher's address
  txnReg :: Signal dom Txn
  txnReg = regEn (Txn 0 0) actualFetchStart (Txn <$> liveRow <*> liveAddr)

  -- ARADDR ASSERTION
  dramRowWithArCheck :: Signal dom (RowI8E ModelDimension)
  dramRowWithArCheck = assertArAddrMatchGeneric
      cycleCounter
      (Layout.dbgArAccepted fetcherDebug)
      (Layout.dbgLatchedAddr fetcherDebug)
      (tAddr <$> txnReg)
      (tRow <$> txnReg)
      tagStr
      dramRowCommitted

  -- Parse and stage the DRAM row
  parsedRow :: Signal dom (RowI8E ModelDimension)
  parsedRow = Layout.multiWordRowParser <$> fetchedWords

  zeroRow :: RowI8E ModelDimension
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  -- Assemble on fetch completion; commit on dvRise (interface contract)
  dramRowAssembled :: Signal dom (RowI8E ModelDimension)
  dramRowAssembled = regEn zeroRow fetchValid parsedRow

  dramRowCommitted :: Signal dom (RowI8E ModelDimension)
  dramRowCommitted = regEn zeroRow dvRise dramRowAssembled

  -- Keep the words for diagnostics
  fetchedWordsAssembled :: Signal dom (Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsAssembled = regEn (repeat 0) fetchValid fetchedWords

  fetchedWordsCommitted :: Signal dom (Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsCommitted = regEn (repeat 0) dvRise fetchedWordsAssembled

  -- HC rows aligned with commit
  hcRowCommitted :: Signal dom (RowI8E ModelDimension)
  hcRowCommitted = regEn zeroRow dvRise ((!!) hcWeights . tRow <$> txnReg)

  -- HC row captured at the same time as the fetch handshake (for assert text)
  hcRowFromCapture :: Signal dom (RowI8E ModelDimension)
  hcRowFromCapture = regEn zeroRow actualFetchStart ((!!) hcWeights <$> liveRow)

  -- Expected address recomputed from the captured row index
  expectedAddrFromRow :: Signal dom (Unsigned 32)
  expectedAddrFromRow =
    Layout.rowAddressCalculator matrixType layerIdx headIdxInt . tRow <$> txnReg

  dramRowWithTrace :: Signal dom (RowI8E ModelDimension)
  dramRowWithTrace = traceRowIndicesGeneric
      cycleCounter
      actualFetchStart
      dvRise
      liveRow
      (tRow <$> txnReg)
      tagStr
      dramRowWithArCheck

  -- COMMIT-TIME CHECK
  dramRowAfterEqCheck :: Signal dom (RowI8E ModelDimension)
  dramRowAfterEqCheck =
    assertRowsMatchOnCommitGeneric
      cycleCounter
      dvRiseD1
      (tRow  <$> txnReg)
      (tAddr <$> txnReg)
      expectedAddrFromRow
      dramRowWithTrace
      hcRowCommitted
      hcRowFromCapture
      fetchedWordsCommitted
      tagStr

  -- Loader FSM next-state
  nextState =
    mux (loadState .==. pure LIdle .&&. actualFetchStart)
        (pure LFetching)
    $ mux (loadState .==. pure LFetching .&&. fetchValid)
        (pure LDone)
    $ mux (loadState .==. pure LDone .&&. downstreamReady .&&. dataConsumed)
        (pure LIdle)
        loadState
  
  -- Outputs
  out :: WeightLoaderOutput dom
  out = WeightLoaderOutput
    { hcRowOut          = hcRowCommitted
    , dramRowOut        = dramRowAfterEqCheck
    , dbgRequestedAddr  = liveAddr
    , dbgCapturedAddr   = tAddr <$> txnReg
    , dbgCapturedRowReq = tRow  <$> txnReg
    , dbgLoadState      = loadState
    }

--------------------------------------------------------------------------------
-- Helper functions (generalized versions)
--------------------------------------------------------------------------------

traceRowIndicesGeneric
  :: Signal dom (Unsigned 32)          -- cycleCounter
  -> Signal dom Bool                   -- actualFetchStart
  -> Signal dom Bool                   -- dvRise
  -> Signal dom (Index HeadDimension)  -- liveRow
  -> Signal dom (Index HeadDimension)  -- txnReg.tRow
  -> String                            -- tag string
  -> Signal dom (RowI8E ModelDimension)  -- pass-through signal
  -> Signal dom (RowI8E ModelDimension)
traceRowIndicesGeneric cyc start rise lRow txnRow tagStr' rowIn =
  go <$> cyc <*> start <*> rise <*> lRow <*> txnRow <*> rowIn
 where
  go c True _ lr _ row = 
    trace ("@" P.++ show c P.++ " " P.++ tagStr' P.++ "reqPulse RISE liveRow=" P.++ show lr) row
  go c _ True _ tr row = 
    let msg = "@" P.++ show c P.++ " " P.++ tagStr' P.++ "COMMIT txnRow=" P.++ show tr
        finalMsg = if tr == maxBound 
                   then msg P.++ " *** FINAL ROW ***"
                   else msg
    in trace finalMsg row
  go _ _ _ _ _ row = row

assertArAddrMatchGeneric
  :: Signal dom (Unsigned 32)
  -> Signal dom Bool
  -> Signal dom (Unsigned 32)
  -> Signal dom (Unsigned 32)
  -> Signal dom (Index HeadDimension)
  -> String
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
assertArAddrMatchGeneric cyc arAccepted fetcherAddr loaderAddr rowIdx tagStr' rowIn =
  check <$> cyc <*> arAccepted <*> fetcherAddr <*> loaderAddr <*> rowIdx <*> rowIn
 where
  check c True fAddr lAddr ri row
    | fAddr /= lAddr = P.error $
        "@" P.++ show c P.++ " ARADDR MISMATCH at AR accept!"
        P.++ "\n  *** LOCATION: " P.++ tagStr' P.++ " row=" P.++ show ri P.++ " ***"
        P.++ "\n  Fetcher latched addr: " P.++ show fAddr
        P.++ "\n  Loader txnReg addr:   " P.++ show lAddr
        P.++ "\n  Delta: " P.++ show (if fAddr > lAddr 
                                       then fAddr - lAddr 
                                       else lAddr - fAddr)
    | otherwise = row
  check _ False _ _ _ row = row

assertRowsMatchOnCommitGeneric
  :: forall dom n. (KnownNat n, KnownNat (If (OrdCond (CmpNat n 63) 'True 'True 'False) 1 (1 + Div n 64) T.* 64))
  => Signal dom (Unsigned 32)
  -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> Signal dom (Unsigned 32)
  -> Signal dom (Unsigned 32)
  -> Signal dom (RowI8E n)
  -> Signal dom (RowI8E n)
  -> Signal dom (RowI8E n)
  -> Signal dom (Vec (Layout.WordsPerRow n) (BitVector 512))
  -> String
  -> Signal dom (RowI8E n)
assertRowsMatchOnCommitGeneric cyc commitEdge capIdx capAddr expectedAddr 
                               dramRow hcRow hcRowCapture fetchedWords' tagStr' =
  mux commitEdge 
      (check <$> cyc <*> capIdx <*> capAddr <*> expectedAddr 
             <*> dramRow <*> hcRow <*> hcRowCapture <*> fetchedWords') 
      dramRow
 where
  check c ri ad expAd dr hr hrc words' =
    let de = rowExponent dr
        he = rowExponent hr
        dm = rowMantissas dr
        hm = rowMantissas hr
        expMatch  = de == he
        mantMatch = dm == hm
        addrMatch = ad == expAd
        nShow = 8 :: Int
        showPrefix xs = P.take nShow (P.map show (toList xs))
        parsedFromWords :: RowI8E ModelDimension
        parsedFromWords = Layout.multiWordRowParser words'
        pw_exp = rowExponent parsedFromWords
        pw_mants = rowMantissas parsedFromWords
    in if expMatch && mantMatch
         then dr
         else
           let dm0 = showPrefix dm
               hm0 = showPrefix hm
               hrc0 = showPrefix (rowMantissas hrc)
               pw0 = showPrefix pw_mants
               word0_hex = case toList words' of
                             (w:_) -> show w
                             _     -> "no words"
           in P.error $
                "@" P.++ show c P.++ " DRAM/HC mismatch at commit!"
             P.++ "\n  *** LOCATION: " P.++ tagStr' P.++ " row=" P.++ show ri P.++ " ***"
             P.++ "\n  capturedAddr=" P.++ show ad
             P.++ "\n  expectedAddr=" P.++ show expAd
             P.++ "\n  ADDR MATCH: " P.++ show addrMatch
             P.++ (if not addrMatch 
                   then "\n  *** ADDRESS MISMATCH - BUG IN CAPTURE TIMING! ***" 
                   else "")
             P.++ "\n  expMatch=" P.++ show expMatch
             P.++ " mantMatch=" P.++ show mantMatch
             P.++ "\n  dramExp=" P.++ show de
             P.++ " hcExp=" P.++ show he
             P.++ "\n  DRAM mant[0..7]=" P.++ show dm0
             P.++ "\n  HC (committed) mant[0..7]=" P.++ show hm0
             P.++ "\n  HC (at capture) mant[0..7]=" P.++ show hrc0
             P.++ "\n  HC rows match: " P.++ show (hr == hrc)
             P.++ "\n  Parsed from fetchedWords exp=" P.++ show pw_exp
             P.++ "\n  Parsed from fetchedWords mant[0..7]=" P.++ show pw0
             P.++ "\n  Raw word[0] (hex)=" P.++ word0_hex

data Txn = Txn
  { tRow  :: Index HeadDimension
  , tAddr :: Unsigned 32
  } deriving (Generic, NFDataX, Show, Eq)
