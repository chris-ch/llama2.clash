module LLaMa2.Layer.Attention.WeightLoader
  ( weightLoader
  , WeightLoaderOutput(..)
  , LoadState(..)
  , assertRowStable
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumQueryHeads, NumLayers )
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
import qualified Simulation.DRAMBackedAxiSlave as Sim
import Clash.Sized.Vector (unsafeFromList)

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

-- | Weight loader with DRAM fetching and hardcoded (HC) reference path.
--
-- == Overview
--
-- This component manages weight loading for a single query head. It provides
-- two parallel paths:
-- 1. __DRAM path__: Fetches weights from external memory via AXI
-- 2. __HC path__: Provides hardcoded reference weights from parameters
--
-- The dual-path design enables verification: DRAM results can be compared
-- against HC results to detect memory or timing errors.
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────────────────┐
--                    │                   weightLoader                          │
--                    │                                                         │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │                 DRAM Path                       │    │
--                    │  │                                                 │    │
--   dramSlaveIn ────►│  │  ┌──────────────┐    ┌──────────────┐           │    │
--     .arready       │  │  │  AXI Master  │    │   Row        │           │    │
--     .rvalid        │  │  │  Controller  │───►│   Assembly   │           │───►│ dramRowOut
--     .rdata         │  │  │              │    │  (2 words→   │           │    │
--                    │  │  │ Issues AR,   │    │   1 row)     │           │    │
--   rowReqValid ────►│  │  │ tracks resp  │    │              │           │    │
--                    │  │  └──────────────┘    └──────────────┘           │    │
--   rowIndex ───────►│  │         │                   │                   │    │
--                    │  │         │                   ▼                   │    │
--   layerIdx ───────►│  │         │           ┌──────────────┐            │    │
--   headIdx ────────►│  │         │           │  DRAM Valid  │────────────│───►│ weightValid
--                    │  │         │           │    Latch     │            │    │
--                    │  │         ▼           └──────────────┘            │    │
--                    │  │  ┌──────────────┐                               │    │
--                    │  │  │  axiMaster   │───────────────────────────────│───►│ axiMaster
--                    │  │  │    Out       │                               │    │
--                    │  │  └──────────────┘                               │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │                  HC Path                        │    │
--                    │  │                                                 │    │
--   params ─────────►│  │  ┌──────────────┐                               │    │
--                    │  │  │   Direct     │                               │───►│ hcRowOut
--                    │  │  │   Lookup     │                               │    │
--                    │  │  │              │                               │    │
--                    │  │  │ qMatrix[head]│                               │    │
--                    │  │  │   [row]      │                               │    │
--                    │  │  └──────────────┘                               │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │              Ready Logic                        │    │
--                    │  │                                                 │    │
--   downStreamReady─►│  │  weightReady = (loaderState == LIdle)           │───►│ weightReady
--                    │  │              || (loaderState == LDone)          │    │
--   rowDone ────────►│  │                                                 │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    └─────────────────────────────────────────────────────────┘
-- @
--
-- == Input Signals
--
-- [@dramSlaveIn@] AXI slave interface from arbiter (routed per-head response).
--                 - arready: Arbiter accepted our address request
--                 - rvalid: Data is available on rdata
--                 - rdata: 512-bit data word from DRAM
--
-- [@layerIdx@] Current transformer layer (0 to NumLayers-1).
--              Used to compute DRAM address offset.
--
-- [@headIdx@] Query head index (0 to NumQueryHeads-1).
--             Used to compute DRAM address offset and HC lookup.
--
-- [@rowIndex@] Current row being processed (0 to HeadDimension-1).
--              Used for both DRAM address and HC lookup.
--
-- [@rowReqValid@] Request signal from multiplier FSM (state == MFetching).
--                 Triggers a new DRAM fetch when loader is idle.
--
-- [@downStreamReady@] Downstream acknowledgment (for state transitions).
--
-- [@rowDone@] Row computation complete signal from multiplier.
--             Used to hold LDone state until row is actually consumed.
--
-- [@params@] Model parameters containing hardcoded weights.
--
-- == Output Signals
--
-- [@axiMaster@] AXI master interface to arbiter.
--               - arvalid: Request to read from DRAM
--               - ardata: Address and burst parameters
--               - rready: Ready to accept read data
--
-- [@weightLoaderOut@] Bundled row outputs.
--                     - dramRowOut: Row assembled from DRAM data
--                     - hcRowOut: Row from hardcoded parameters
--
-- [@weightValid@] __Level signal__. True when DRAM row is valid and stable.
--                 The multiplier should only process when this is True.
--
-- [@weightReady@] __Level signal__. True when loader can accept new request.
--                 Used to gate rowReqValid from the multiplier.
--
-- == Loader State Machine
--
-- @
--     ┌─────────┐  rowReqValid    ┌───────────────┐
--     │  LIdle  │────────────────►│ LFetchingWord1│
--     │         │                 │               │
--     │ ready=T │                 │ Issue AR for  │
--     │ valid=F │                 │ first 512-bit │
--     └────┬────┘                 │ word          │
--          ▲                      └───────┬───────┘
--          │                              │ rvalid (word 1 arrives)
--          │                              ▼
--          │                      ┌───────────────┐
--          │                      │ LFetchingWord2│
--          │                      │               │
--          │                      │ Issue AR for  │
--          │                      │ second 512-bit│
--          │                      │ word          │
--          │                      └───────┬───────┘
--          │                              │ rvalid (word 2 arrives)
--          │                              ▼
--          │                      ┌───────────────┐
--          │  downStreamReady     │    LDone      │
--          │  && rowDone          │               │
--          └──────────────────────│ ready=T       │
--                                 │ valid=T       │
--                                 │               │
--                                 │ (holds until  │
--                                 │  consumed)    │
--                                 └───────────────┘
-- @
--
-- == Memory Layout
--
-- Each row requires 2 DRAM words (2 × 512 bits = 128 bytes):
-- - Word 1: mantissas[0..63] (64 × 8-bit = 512 bits)
-- - Word 2: mantissas[0] (8-bit) + exponent (8-bit) + padding
--
-- Note: For ModelDimension=64, one row fits in ~65 bytes, but aligned to 128.
--
-- Address calculation:
-- @
-- baseAddr = weightsBaseOffset 
--          + layerIdx * layerStride
--          + qWeightsOffset
--          + headIdx * headStride
--          + rowIndex * rowStride
--
-- word1Addr = baseAddr
-- word2Addr = baseAddr + 64  -- Next 512-bit word
-- @
--
-- == Row Assembly
--
-- @
-- ┌─────────────────────────────────────────────────────────────────┐
-- │                        DRAM Word 1 (512 bits)                   │
-- │  ┌──────┬──────┬──────┬─────────────────────────────┬──────┐    │
-- │  │mant0 │mant1 │mant2 │  ...                        │mant63│    │
-- │  │ 8b   │ 8b   │ 8b   │                             │ 8b   │    │
-- │  └──────┴──────┴──────┴─────────────────────────────┴──────┘    │
-- └─────────────────────────────────────────────────────────────────┘
--
-- ┌─────────────────────────────────────────────────────────────────┐
-- │                        DRAM Word 2 (512 bits)                   │
-- │  ┌──────┬──────┬──────────────────────────────────────────────┐ │
-- │  │ exp  │ pad  │              (unused)                        │ │
-- │  │ 8b   │      │                                              │ │
-- │  └──────┴──────┴──────────────────────────────────────────────┘ │
-- └─────────────────────────────────────────────────────────────────┘
--
--                              │
--                              ▼ Assembly
--
-- ┌─────────────────────────────────────────────────────────────────┐
-- │                      RowI8E ModelDimension                      │
-- │  rowMantissas: Vec 64 (Signed 8) = [mant0, mant1, ..., mant63]  │
-- │  rowExponent:  Signed 16         = sign-extended exp            │
-- └─────────────────────────────────────────────────────────────────┘
-- @
--
-- == Timing Diagram
--
-- @
-- Cycle:         0    1    2    3    4    5    6    7    8
--
-- rowReqValid:   ─────┐___________________________________________
-- loaderState:   Idle Idle Ftch1 Ftch1 Ftch2 Ftch2 Done Done Idle
-- 
-- axiMaster.arv: ___________┐_____┐_____┐_________________________
-- slaveIn.ardy:  ___________┐_____┐___________________________________
-- slaveIn.rvalid:______________┐________┐____________________________
-- 
-- word1Latched:  _______________┐────────────────────────────────────
-- word2Latched:  ______________________┐─────────────────────────────
-- 
-- weightValid:   ______________________┐────────────────┐____________
-- weightReady:   ───────────┐________________________┐_______________
--
-- rowDone:       ____________________________________┐________________
-- downStreamRdy: ____________________________________┐________________
-- @
--
-- == HC Path Operation
--
-- The hardcoded path is purely combinational:
--
-- @
-- hcRowOut = qMatrix params !! layerIdx !! headIdx !! rowIndex
-- @
--
-- This provides a reference value that tracks rowIndex changes immediately,
-- while DRAM path has multi-cycle latency. The comparison between paths
-- happens in queryHeadMatrixMultiplier after both paths have valid data.
--
-- == Stability Assertion
--
-- The loader includes an assertion helper:
--
-- @
-- assertRowStable :: Signal dom Bool -> Signal dom (RowI8E n) -> Signal dom (RowI8E n)
-- assertRowStable valid row = ...
-- @
--
-- This checks that committed row data doesn't change while valid=True,
-- catching bugs where DRAM data might shift unexpectedly.
--
-- == Integration with Multiplier
--
-- @
-- -- In queryHeadMatrixMultiplier:
-- 
-- (axiMaster, weightLoaderOut, weightValid, weightReady) =
--     LOADER.weightLoader dramSlaveIn layerIdx headIdx
--                         rowIndex rowReqValidGated downStreamReady
--                         (moRowDone multOut)
--                         params
--
-- -- Gate request by ready to prevent spurious requests
-- rowReqValidGated = moRowReqValid multOut .&&. weightReady
--
-- -- Pass valid to multiplier FSM (rowValid input)
-- multOut = multiplier xHat currentRowHC inputValidLatched' weightValid ...
-- @
--
-- == Usage Notes
--
-- 1. Always gate rowReqValid with weightReady to prevent requests while busy.
--
-- 2. The loader holds LDone until rowDone fires, ensuring data stability
--    throughout the row computation.
--
-- 3. For HC-only operation, weightValid can be driven by (loaderState == LDone)
--    or simply pure True if no DRAM latency simulation is needed.
--
-- 4. The DRAM and HC paths should produce identical results; any mismatch
--    indicates a bug in address calculation, data assembly, or timing.
--
weightLoader :: forall dom. HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
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
weightLoader dram layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed params =
  (axiMaster, out, weightValid, weightReady)
 where
  -- Hardcoded (HC) weights for this layer/head
  hcWeights :: MatI8E HeadDimension ModelDimension
  hcWeights =
    PARAM.qMatrix
      (PARAM.qHeads (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)

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
  liveRow  :: Signal dom (Index HeadDimension)
  liveRow  = rowReq

  liveAddr :: Signal dom (Unsigned 32)
  liveAddr = Layout.rowAddressCalculator Layout.QMatrix layerIdx headIdx <$> liveRow

  -- ========== KEY FIX: Use fetcher's ready as the single handshake ==========
  -- The actual fetch start: requires BOTH loader idle AND fetcher ready
  actualFetchStart :: Signal dom Bool
  actualFetchStart = weightReady .&&. fetcherReady .&&. rowReqValid

  -- AXI multi-word fetcher - now returns debug signals too
  (axiMaster, fetchedWords, fetchValid, fetcherReady, fetcherDebug) =
    Layout.axiMultiWordRowFetcher @_ @ModelDimension dram actualFetchStart liveAddr

  -- Transaction record captured on THE SAME handshake as the fetcher's address
  txnReg :: Signal dom Txn
  txnReg = regEn (Txn 0 0) actualFetchStart (Txn <$> liveRow <*> liveAddr)
  -- ========== END KEY FIX ==========

  -- ========== ARADDR ASSERTION ==========
  -- Verify that when AR is accepted, the fetcher's latched address matches our txnReg
  dramRowWithArCheck :: Signal dom (RowI8E ModelDimension)
  dramRowWithArCheck = assertArAddrMatch
      (Layout.dbgArAccepted fetcherDebug)
      (Layout.dbgLatchedAddr fetcherDebug)
      (tAddr <$> txnReg)
      (tRow <$> txnReg)
      layerIdx
      headIdx
      dramRowCommitted  -- pass through this signal
  -- ========== END ARADDR ASSERTION ==========

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
    Layout.rowAddressCalculator Layout.QMatrix layerIdx headIdx . tRow <$> txnReg

  dramRowWithTrace :: Signal dom (RowI8E ModelDimension)
  dramRowWithTrace = traceRowIndices
      actualFetchStart
      dvRise
      liveRow
      (tRow <$> txnReg)
      headIdx              -- add this parameter
      dramRowWithArCheck

  -- COMMIT-TIME CHECK
  dramRowAfterEqCheck :: Signal dom (RowI8E ModelDimension)
  dramRowAfterEqCheck =
    assertRowsMatchOnCommitEnhanced
      dvRiseD1
      (tRow  <$> txnReg)
      (tAddr <$> txnReg)
      expectedAddrFromRow
      dramRowWithTrace    -- changed
      hcRowCommitted
      hcRowFromCapture
      fetchedWordsCommitted
      layerIdx
      headIdx

  -- Loader FSM next-state - use actualFetchStart for consistency
  nextState =
    mux (loadState .==. pure LIdle .&&. actualFetchStart)
        (pure LFetching)
    $ mux (loadState .==. pure LFetching .&&. fetchValid)
        (pure LDone)
    $ mux (loadState .==. pure LDone .&&. downstreamReady .&&. dataConsumed)
        (pure LIdle)
        loadState

  dramRowFinal :: Signal dom (RowI8E ModelDimension)
  dramRowFinal = staticDramCheck params layerIdx headIdx hcWeights dramRowAfterEqCheck
  
  -- Outputs
  out :: WeightLoaderOutput dom
  out = WeightLoaderOutput
    { hcRowOut          = hcRowCommitted
    , dramRowOut        = dramRowFinal
    , dbgRequestedAddr  = liveAddr
    , dbgCapturedAddr   = tAddr <$> txnReg
    , dbgCapturedRowReq = tRow  <$> txnReg
    , dbgLoadState      = loadState
    }

staticDramCheck
  :: PARAM.DecoderParameters
  -> Index NumLayers
  -> Index NumQueryHeads
  -> MatI8E HeadDimension ModelDimension  -- hcWeights
  -> Signal dom a  -- pass-through
  -> Signal dom a
staticDramCheck params layerIdx' headIdx' hcW sig =
  let dramImage = Sim.buildMemoryFromParams @65536 params
      -- Check row 0
      testRow = 0 :: Index HeadDimension
      testAddr = Layout.rowAddressCalculator Layout.QMatrix layerIdx' headIdx' testRow
      wordIdx = fromIntegral testAddr `div` 64 :: Int
      numWords = Layout.wordsPerRowVal @ModelDimension
      dramWordsList = P.take numWords (P.drop wordIdx (toList dramImage))
      dramRow = Layout.multiWordRowParser 
                  (unsafeFromList dramWordsList 
                     :: Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
      hcRow = hcW !! testRow
      
      errMsg = "STATIC CHECK FAILED!"
            P.++ "\n  layer=" P.++ show layerIdx'
            P.++ " head=" P.++ show headIdx'
            P.++ " row=" P.++ show testRow
            P.++ "\n  addr=" P.++ show testAddr
            P.++ " wordIdx=" P.++ show wordIdx
            P.++ "\n  DRAM exp=" P.++ show (rowExponent dramRow)
            P.++ " HC exp=" P.++ show (rowExponent hcRow)
            P.++ "\n  DRAM[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas dramRow)
            P.++ "\n  HC[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas hcRow)
  in if dramRow /= hcRow
     then P.error errMsg
     else sig

traceRowIndices
  :: Signal dom Bool                   -- actualFetchStart
  -> Signal dom Bool                   -- dvRise
  -> Signal dom (Index HeadDimension)  -- liveRow
  -> Signal dom (Index HeadDimension)  -- txnReg.tRow
  -> Index NumQueryHeads               -- headIdx (static)
  -> Signal dom (RowI8E ModelDimension)  -- pass-through signal
  -> Signal dom (RowI8E ModelDimension)
traceRowIndices start rise lRow txnRow headIdx' rowIn =
  go <$> start <*> rise <*> lRow <*> txnRow <*> rowIn
 where
  go True _ lr _ row = 
    trace ("CAPTURE H" P.++ show headIdx' P.++ ": liveRow=" P.++ show lr) row
  go _ True _ tr row = 
    let msg = "COMMIT H" P.++ show headIdx' P.++ ": txnReg.tRow=" P.++ show tr
        finalMsg = if tr == maxBound 
                   then msg P.++ " *** FINAL ROW ***"
                   else msg
    in trace finalMsg row
  go _ _ _ _ row = row

-- | ARADDR assertion: fires when AR is accepted if addresses don't match
-- Change assertArAddrMatch to return the row unchanged (passthrough) but check on the way
assertArAddrMatch
  :: Signal dom Bool                   -- ^ arAccepted (from fetcher)
  -> Signal dom (Unsigned 32)          -- ^ fetcher's latched address
  -> Signal dom (Unsigned 32)          -- ^ loader's txnReg address
  -> Signal dom (Index HeadDimension)  -- ^ row index for diagnostics
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (RowI8E ModelDimension)  -- ^ row to pass through
  -> Signal dom (RowI8E ModelDimension)  -- ^ same row, but assertion checked
assertArAddrMatch arAccepted fetcherAddr loaderAddr rowIdx layerIdx' headIdx' rowIn =
  check <$> arAccepted <*> fetcherAddr <*> loaderAddr <*> rowIdx <*> rowIn
 where
  check True fAddr lAddr ri row
    | fAddr /= lAddr = P.error $
        "ARADDR MISMATCH at AR accept!"
        P.++ "\n  *** LOCATION: layer=" P.++ show layerIdx'
        P.++ " head=" P.++ show headIdx'
        P.++ " row=" P.++ show ri P.++ " ***"
        P.++ "\n  Fetcher latched addr: " P.++ show fAddr
        P.++ "\n  Loader txnReg addr:   " P.++ show lAddr
        P.++ "\n  Delta: " P.++ show (if fAddr > lAddr 
                                       then fAddr - lAddr 
                                       else lAddr - fAddr)
    | otherwise = row
  check False _ _ _ row = row

assertRowsMatchOnCommitEnhanced 
  :: forall dom n. (KnownNat n, KnownNat (If (OrdCond (CmpNat n 63) 'True 'True 'False) 1 (1 + Div n 64) T.* 64))
  => Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> Signal dom (Unsigned 32)
  -> Signal dom (Unsigned 32)
  -> Signal dom (RowI8E n)
  -> Signal dom (RowI8E n)
  -> Signal dom (RowI8E n)
  -> Signal dom (Vec (Layout.WordsPerRow n) (BitVector 512))
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (RowI8E n)
assertRowsMatchOnCommitEnhanced commitEdge capIdx capAddr expectedAddr 
                                 dramRow hcRow hcRowCapture fetchedWords' 
                                 layerIdx' headIdx' =
  mux commitEdge 
      (check <$> capIdx <*> capAddr <*> expectedAddr 
             <*> dramRow <*> hcRow <*> hcRowCapture <*> fetchedWords') 
      dramRow
 where
  check ri ad expAd dr hr hrc words' =
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
                "DRAM/HC mismatch at commit!"
             P.++ "\n  *** LOCATION: layerIdx=" P.++ show layerIdx' 
             P.++ " headIdx=" P.++ show headIdx' 
             P.++ " row=" P.++ show ri P.++ " ***"
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
