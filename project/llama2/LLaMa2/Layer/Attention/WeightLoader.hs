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
-- Now takes cycleCounter for cycle-stamped tracing.
--
weightLoader :: forall dom. HiddenClockResetEnable dom
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
weightLoader cycleCounter dram layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed params =
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
      cycleCounter
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
      cycleCounter
      actualFetchStart
      dvRise
      liveRow
      (tRow <$> txnReg)
      layerIdx
      headIdx
      dramRowWithArCheck

  -- COMMIT-TIME CHECK
  dramRowAfterEqCheck :: Signal dom (RowI8E ModelDimension)
  dramRowAfterEqCheck =
    assertRowsMatchOnCommitEnhanced
      cycleCounter
      dvRiseD1
      (tRow  <$> txnReg)
      (tAddr <$> txnReg)
      expectedAddrFromRow
      dramRowWithTrace
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

-- | Trace weightReady edges
traceWeightReady
  :: forall dom. HiddenClockResetEnable dom => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool
  -> Signal dom Bool
traceWeightReady cyc layerIdx' headIdx' sig = result
  where
    prev :: Signal dom Bool
    prev = register False sig
    result = emit <$> cyc <*> sig <*> prev
    emit c True False = trace ("@" P.++ show c P.++ " [WFU L" P.++ show layerIdx' 
                               P.++ " H" P.++ show headIdx' P.++ "] weightReady RISE") True
    emit c False True = trace ("@" P.++ show c P.++ " [WFU L" P.++ show layerIdx' 
                               P.++ " H" P.++ show headIdx' P.++ "] weightReady FALL") False
    emit _ curr _ = curr

-- | Trace weightValid edges
traceWeightValid
  :: forall dom. HiddenClockResetEnable dom => Signal dom (Unsigned 32)
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool
  -> Signal dom Bool
traceWeightValid cyc layerIdx' headIdx' sig = result
  where
    prev = register False sig
    result = emit <$> cyc <*> sig <*> prev
    emit c True False = trace ("@" P.++ show c P.++ " [WFU L" P.++ show layerIdx' 
                               P.++ " H" P.++ show headIdx' P.++ "] weightValid RISE") True
    emit c False True = trace ("@" P.++ show c P.++ " [WFU L" P.++ show layerIdx' 
                               P.++ " H" P.++ show headIdx' P.++ "] weightValid FALL") False
    emit _ curr _ = curr

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
  :: Signal dom (Unsigned 32)          -- cycleCounter
  -> Signal dom Bool                   -- actualFetchStart
  -> Signal dom Bool                   -- dvRise
  -> Signal dom (Index HeadDimension)  -- liveRow
  -> Signal dom (Index HeadDimension)  -- txnReg.tRow
  -> Index NumLayers                   -- layerIdx (static)
  -> Index NumQueryHeads               -- headIdx (static)
  -> Signal dom (RowI8E ModelDimension)  -- pass-through signal
  -> Signal dom (RowI8E ModelDimension)
traceRowIndices cyc start rise lRow txnRow layerIdx' headIdx' rowIn =
  go <$> cyc <*> start <*> rise <*> lRow <*> txnRow <*> rowIn
 where
  go c True _ lr _ row = 
    trace ("@" P.++ show c P.++ " [WFU L" P.++ show layerIdx' P.++ " H" P.++ show headIdx' 
           P.++ "] reqPulse RISE liveRow=" P.++ show lr) row
  go c _ True _ tr row = 
    let msg = "@" P.++ show c P.++ " [WFU L" P.++ show layerIdx' P.++ " H" P.++ show headIdx' 
              P.++ "] COMMIT txnRow=" P.++ show tr
        finalMsg = if tr == maxBound 
                   then msg P.++ " *** FINAL ROW ***"
                   else msg
    in trace finalMsg row
  go _ _ _ _ _ row = row

-- | ARADDR assertion: fires when AR is accepted if addresses don't match
assertArAddrMatch
  :: Signal dom (Unsigned 32)          -- ^ cycleCounter
  -> Signal dom Bool                   -- ^ arAccepted (from fetcher)
  -> Signal dom (Unsigned 32)          -- ^ fetcher's latched address
  -> Signal dom (Unsigned 32)          -- ^ loader's txnReg address
  -> Signal dom (Index HeadDimension)  -- ^ row index for diagnostics
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (RowI8E ModelDimension)  -- ^ row to pass through
  -> Signal dom (RowI8E ModelDimension)  -- ^ same row, but assertion checked
assertArAddrMatch cyc arAccepted fetcherAddr loaderAddr rowIdx layerIdx' headIdx' rowIn =
  check <$> cyc <*> arAccepted <*> fetcherAddr <*> loaderAddr <*> rowIdx <*> rowIn
 where
  check c True fAddr lAddr ri row
    | fAddr /= lAddr = P.error $
        "@" P.++ show c P.++ " ARADDR MISMATCH at AR accept!"
        P.++ "\n  *** LOCATION: layer=" P.++ show layerIdx'
        P.++ " head=" P.++ show headIdx'
        P.++ " row=" P.++ show ri P.++ " ***"
        P.++ "\n  Fetcher latched addr: " P.++ show fAddr
        P.++ "\n  Loader txnReg addr:   " P.++ show lAddr
        P.++ "\n  Delta: " P.++ show (if fAddr > lAddr 
                                       then fAddr - lAddr 
                                       else lAddr - fAddr)
    | otherwise = row
  check _ False _ _ _ row = row

assertRowsMatchOnCommitEnhanced 
  :: forall dom n. (KnownNat n, KnownNat (If (OrdCond (CmpNat n 63) 'True 'True 'False) 1 (1 + Div n 64) T.* 64))
  => Signal dom (Unsigned 32)  -- cycleCounter
  -> Signal dom Bool
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
assertRowsMatchOnCommitEnhanced cyc commitEdge capIdx capAddr expectedAddr 
                                 dramRow hcRow hcRowCapture fetchedWords' 
                                 layerIdx' headIdx' =
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
