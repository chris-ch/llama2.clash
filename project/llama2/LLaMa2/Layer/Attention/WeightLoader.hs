module LLaMa2.Layer.Attention.WeightLoader
  ( qWeightLoader      -- Q weights
  , kWeightLoader      -- K weights
  , vWeightLoader      -- V weights
  , woWeightLoader     -- WO output-projection weights
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

-- | Output bundle from a weight loader.  'numCols' is the column dimension
-- of the weight matrix (ModelDimension for Q/K/V, HeadDimension for WO).
data WeightLoaderOutput dom numCols = WeightLoaderOutput
  { hcRowOut          :: Signal dom (RowI8E numCols)
  , dramRowOut        :: Signal dom (RowI8E numCols)
  , dbgRequestedAddr  :: Signal dom (Unsigned 32)
  , dbgCapturedAddr   :: Signal dom (Unsigned 32)
  , dbgCapturedRowReq :: Signal dom Int             -- row index as Int
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

-- | Weight loader for Q matrices (numRows=HeadDimension, numCols=ModelDimension)
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
     , WeightLoaderOutput dom ModelDimension
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

-- | Weight loader for K matrices (numRows=HeadDimension, numCols=ModelDimension)
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
     , WeightLoaderOutput dom ModelDimension
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

-- | Weight loader for V matrices (numRows=HeadDimension, numCols=ModelDimension)
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
     , WeightLoaderOutput dom ModelDimension
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

-- | Weight loader for WO output-projection matrices.
-- WO is transposed vs Q/K/V: numRows=ModelDimension (64 rows), numCols=HeadDimension (8 cols).
woWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads            -- ^ Q-head index (WO has NumQueryHeads heads)
  -> Signal dom (Index ModelDimension)  -- ^ row request (0..ModelDimension-1)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom HeadDimension
     , Signal dom Bool
     , Signal dom Bool
     )
woWeightLoader cycleCounter dram layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed params =
  weightLoaderGeneric cycleCounter dram Layout.WOMatrix layerIdx (fromIntegral headIdx)
                      rowReq rowReqValid downstreamReady dataConsumed hcWeights tagStr
 where
  hcWeights :: MatI8E ModelDimension HeadDimension
  hcWeights = PARAM.mWoQ (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx
  tagStr = "[WOWFU L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ "] "

-- | Generic weight loader for any matrix type and dimensions.
--
-- Type parameters:
--   numRows — number of rows in the weight matrix (= HeadDimension for Q/K/V, ModelDimension for WO)
--   numCols — number of columns                   (= ModelDimension for Q/K/V, HeadDimension for WO)
--
-- Existing Q/K/V callers resolve (numRows=HeadDimension, numCols=ModelDimension) via type inference.
weightLoaderGeneric :: forall dom numRows numCols.
  ( HiddenClockResetEnable dom
  , KnownNat numRows
  , KnownNat numCols
  , KnownNat (Layout.WordsPerRow numCols)
  , KnownNat (If (OrdCond (CmpNat numCols 63) 'True 'True 'False) 1 (1 + Div numCols 64) T.* 64)
  )
  => Signal dom (Unsigned 32)       -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom
  -> Layout.MatrixType              -- ^ Which matrix (QMatrix, KMatrix, VMatrix, WOMatrix)
  -> Index NumLayers
  -> Int                            -- ^ Head index as Int
  -> Signal dom (Index numRows)     -- ^ Row request
  -> Signal dom Bool                -- ^ rowReqValid (pulse)
  -> Signal dom Bool                -- ^ downstreamReady (level)
  -> Signal dom Bool                -- ^ dataConsumed (pulse)
  -> MatI8E numRows numCols         -- ^ HC weights for cross-check
  -> String                         -- ^ Tag string for tracing
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom numCols
     , Signal dom Bool              -- ^ weightValid (level)
     , Signal dom Bool              -- ^ weightReady (level)
     )
weightLoaderGeneric cycleCounter dram matrixType layerIdx headIdxInt rowReq rowReqValid downstreamReady dataConsumed hcWeights tagStr =
  (axiMaster, out, weightValid, weightReady)
 where
  -- Loader FSM
  loadState :: Signal dom LoadState
  loadState = register LIdle nextState

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
  liveRow  :: Signal dom (Index numRows)
  liveRow  = rowReq

  liveAddr :: Signal dom (Unsigned 32)
  liveAddr = Layout.rowAddressCalculator matrixType layerIdx headIdxInt <$> (fromIntegral <$> liveRow)

  -- The actual fetch start: requires BOTH loader idle AND fetcher ready
  actualFetchStart :: Signal dom Bool
  actualFetchStart = weightReady .&&. fetcherReady .&&. rowReqValid

  -- AXI multi-word fetcher
  (axiMaster, fetchedWords, fetchValid, fetcherReady, fetcherDebug) =
    Layout.axiMultiWordRowFetcher @_ @numCols dram actualFetchStart liveAddr

  -- Transaction record captured on THE SAME handshake as the fetcher's address
  txnReg :: Signal dom (Txn numRows)
  txnReg = regEn (Txn 0 0) actualFetchStart (Txn <$> liveRow <*> liveAddr)

  -- ARADDR ASSERTION
  dramRowWithArCheck :: Signal dom (RowI8E numCols)
  dramRowWithArCheck = assertArAddrMatchGeneric
      cycleCounter
      (Layout.dbgArAccepted fetcherDebug)
      (Layout.dbgLatchedAddr fetcherDebug)
      (tAddr <$> txnReg)
      (tRow <$> txnReg)
      tagStr
      dramRowCommitted

  -- Parse and stage the DRAM row
  parsedRow :: Signal dom (RowI8E numCols)
  parsedRow = Layout.multiWordRowParser <$> fetchedWords

  zeroRow :: RowI8E numCols
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  -- Assemble on fetch completion; commit on dvRise (interface contract)
  dramRowAssembled :: Signal dom (RowI8E numCols)
  dramRowAssembled = regEn zeroRow fetchValid parsedRow

  dramRowCommitted :: Signal dom (RowI8E numCols)
  dramRowCommitted = regEn zeroRow dvRise dramRowAssembled

  -- Keep the words for diagnostics
  fetchedWordsAssembled :: Signal dom (Vec (Layout.WordsPerRow numCols) (BitVector 512))
  fetchedWordsAssembled = regEn (repeat 0) fetchValid fetchedWords

  fetchedWordsCommitted :: Signal dom (Vec (Layout.WordsPerRow numCols) (BitVector 512))
  fetchedWordsCommitted = regEn (repeat 0) dvRise fetchedWordsAssembled

  -- HC rows aligned with commit
  hcRowCommitted :: Signal dom (RowI8E numCols)
  hcRowCommitted = regEn zeroRow dvRise ((!!) hcWeights . tRow <$> txnReg)

  -- HC row captured at the same time as the fetch handshake (for assert text)
  hcRowFromCapture :: Signal dom (RowI8E numCols)
  hcRowFromCapture = regEn zeroRow actualFetchStart ((!!) hcWeights <$> liveRow)

  -- Expected address recomputed from the captured row index
  expectedAddrFromRow :: Signal dom (Unsigned 32)
  expectedAddrFromRow =
    Layout.rowAddressCalculator matrixType layerIdx headIdxInt . fromIntegral . tRow <$> txnReg

  dramRowWithTrace :: Signal dom (RowI8E numCols)
  dramRowWithTrace = traceRowIndicesGeneric
      cycleCounter
      actualFetchStart
      dvRise
      liveRow
      (tRow <$> txnReg)
      tagStr
      dramRowWithArCheck

  -- COMMIT-TIME CHECK
  dramRowAfterEqCheck :: Signal dom (RowI8E numCols)
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
  out :: WeightLoaderOutput dom numCols
  out = WeightLoaderOutput
    { hcRowOut          = hcRowCommitted
    , dramRowOut        = dramRowAfterEqCheck
    , dbgRequestedAddr  = liveAddr
    , dbgCapturedAddr   = tAddr <$> txnReg
    , dbgCapturedRowReq = fromIntegral . tRow <$> txnReg
    , dbgLoadState      = loadState
    }

--------------------------------------------------------------------------------
-- Helper functions (generalized over numRows / numCols)
--------------------------------------------------------------------------------

traceRowIndicesGeneric :: forall dom numRows numCols. KnownNat numRows
  => Signal dom (Unsigned 32)           -- cycleCounter
  -> Signal dom Bool                    -- actualFetchStart
  -> Signal dom Bool                    -- dvRise
  -> Signal dom (Index numRows)         -- liveRow
  -> Signal dom (Index numRows)         -- txnReg.tRow
  -> String                             -- tag string
  -> Signal dom (RowI8E numCols)        -- pass-through signal
  -> Signal dom (RowI8E numCols)
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

assertArAddrMatchGeneric :: forall dom numRows numCols.
     Signal dom (Unsigned 32)
  -> Signal dom Bool
  -> Signal dom (Unsigned 32)
  -> Signal dom (Unsigned 32)
  -> Signal dom (Index numRows)
  -> String
  -> Signal dom (RowI8E numCols)
  -> Signal dom (RowI8E numCols)
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
  :: forall dom numRows n.
     ( KnownNat n
     , KnownNat (If (OrdCond (CmpNat n 63) 'True 'True 'False) 1 (1 + Div n 64) T.* 64)
     )
  => Signal dom (Unsigned 32)
  -> Signal dom Bool
  -> Signal dom (Index numRows)
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
        parsedFromWords :: RowI8E n
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

data Txn numRows = Txn
  { tRow  :: Index numRows
  , tAddr :: Unsigned 32
  } deriving (Generic, NFDataX, Show, Eq)
