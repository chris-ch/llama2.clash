module LLaMa2.Layer.Attention.WeightLoader
  ( weightLoader
  , WeightLoaderOutput(..)
  , LoadState(..)
  , assertRowStable          -- optional, for reuse at the consumer
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, NumQueryHeads, NumLayers )
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightStreaming as STREAM
import qualified Simulation.Parameters as PARAM
import qualified Prelude as P
import Data.Type.Bool (If)
import Data.Type.Ord (OrdCond)
import qualified GHC.TypeNats as T

data LoadState = LIdle | LFetching | LDone
  deriving (Show, Eq, Generic, NFDataX)

data WeightLoaderOutput dom = WeightLoaderOutput
  { hcRowOut          :: Signal dom (RowI8E ModelDimension)
  , dramRowOut        :: Signal dom (RowI8E ModelDimension)
  , dbgRequestedAddr  :: Signal dom (Unsigned 32)  -- live addr (combinational from rowReq)
  , dbgLoadState      :: Signal dom LoadState
  , dbgFetchTrigger   :: Signal dom Bool
  , dbgMultiWordValid :: Signal dom Bool
  , dbgFetchedWords   :: Signal dom (Vec (STREAM.WordsPerRow ModelDimension) (BitVector 512))
  , dbgCapturedRowReq :: Signal dom (Index HeadDimension)
  , dbgCapturedAddr   :: Signal dom (Unsigned 32)
  }

rowStrideBytesI8E :: forall n. KnownNat n => Int
rowStrideBytesI8E = STREAM.wordsPerRowVal @n * 64

-- Simulation-only data-path checker: row does not change while 'validSig' is high
assertRowStable
  :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
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

weightLoader
  :: forall dom. HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)  -- rowReq
  -> Signal dom Bool                   -- rowReqValid (from consumer FSM)
  -> Signal dom Bool                   -- downstreamReady (consumer backpressure)
  -> Signal dom Bool                   -- dataConsumed (rowDone)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool                 -- weightValid (level: LDone)
     , Signal dom Bool                 -- weightReady (level: LIdle)
     )
weightLoader dramSlaveIn layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed params =
  (axiMaster, loaderOutput, weightValidLevel, weightReadyLevel)
 where
  -- Hardcoded weights for the selected head
  hcWeights :: MatI8E HeadDimension ModelDimension
  hcWeights =
    PARAM.wqHeadQ
      (PARAM.headsQ (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)

  -- FSM: Idle -> Fetching -> Done (exit Done only when downstreamReady && dataConsumed)
  loadState :: Signal dom LoadState
  loadState = register LIdle nextState
   where
    nextState =
      withLoad <$> loadState <*> rowReqValid <*> fetchValid <*> downstreamReady <*> dataConsumed
    withLoad LIdle     rv _  _  _  = if rv then LFetching else LIdle
    withLoad LFetching _  fv _  _  = if fv then LDone     else LFetching
    withLoad LDone     _  _  rd dc = if rd && dc then LIdle else LDone

  weightReadyLevel :: Signal dom Bool
  weightReadyLevel = loadState .==. pure LIdle

  weightValidLevel :: Signal dom Bool
  weightValidLevel = loadState .==. pure LDone

  -- Rising edge when the row becomes visible to the consumer (entering LDone)
  prevValid = register False weightValidLevel
  dvRise    = weightValidLevel .&&. (not <$> prevValid)

  -- Reset-safe fetch trigger (FIXED VERSION)
  outOfReset     = register False (pure True)
  outOfResetPrev = register False outOfReset

  -- Capture both the flag AND the row index during reset
  reqDuringResetFlag :: Signal dom Bool
  reqDuringResetFlag = register False (rowReqValid .&&. weightReadyLevel)

  reqDuringResetRow :: Signal dom (Index HeadDimension)
  reqDuringResetRow = regEn 0 (rowReqValid .&&. weightReadyLevel) rowReq

  replayFirst :: Signal dom Bool
  replayFirst = (not <$> outOfResetPrev) .&&. outOfReset .&&. reqDuringResetFlag

  fetchTrigger :: Signal dom Bool
  fetchTrigger =
    (rowReqValid .&&. weightReadyLevel .&&. outOfReset) .||. replayFirst

  -- Use the captured row index when replaying, otherwise use live rowReq
  rowReqEffective :: Signal dom (Index HeadDimension)
  rowReqEffective = mux replayFirst reqDuringResetRow rowReq

  -- Live address from the EFFECTIVE row (uses captured row during replay)
  liveRowAddr :: Signal dom (Unsigned 32)
  liveRowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowReqEffective

  capturedAddr :: Signal dom (Unsigned 32)
  capturedAddr = register 0 $ mux fetchTrigger liveRowAddr capturedAddr

  -- AXI multiword fetcher + parser
  (axiMaster, fetchedWords, fetchValid, _requestReady) =
      STREAM.axiMultiWordRowFetcher @_ @ModelDimension dramSlaveIn fetchTrigger liveRowAddr

  parsedRow :: Signal dom (RowI8E ModelDimension)
  parsedRow = STREAM.multiWordRowParser <$> fetchedWords

  -- Stage at "row assembled"
  zeroRow :: RowI8E ModelDimension
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  dramRowAssembled :: Signal dom (RowI8E ModelDimension)
  dramRowAssembled = regEn zeroRow fetchValid parsedRow

  -- Commit both rows on dvRise (entering LDone)
  dramRowCommitted :: Signal dom (RowI8E ModelDimension)
  dramRowCommitted = regEn zeroRow dvRise dramRowAssembled

  capturedRowReq :: Signal dom (Index HeadDimension)
  capturedRowReq = register 0 $ mux fetchTrigger rowReqEffective capturedRowReq

  -- Capture HC row at fetch time, hold through commit pipeline
  hcRowAtAssemble :: Signal dom (RowI8E ModelDimension)
  hcRowAtAssemble = regEn zeroRow fetchValid ((!!) hcWeights <$> capturedRowReq)

  hcRowCommitted :: Signal dom (RowI8E ModelDimension)
  hcRowCommitted = regEn zeroRow dvRise hcRowAtAssemble

  -- Stage 1: Capture when fetch completes
  fetchedWordsAssembled :: Signal dom (Vec (STREAM.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsAssembled = regEn (repeat 0) fetchValid fetchedWords

  -- Stage 2: Commit when dvRise fires (same as dramRowCommitted)
  fetchedWordsCommitted :: Signal dom (Vec (STREAM.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsCommitted = regEn (repeat 0) dvRise fetchedWordsAssembled

  -- Use committed words in assertion
  dramRowAfterEqCheck = assertRowsMatchOnCommit dvRise capturedRowReq capturedAddr
    dramRowCommitted hcRowCommitted fetchedWordsCommitted

  -- OPTIONAL: sanity-check the row stride between sequential commits (detects +64B bug)
  -- This is on the live path (no DCE) but cheap. Safe to keep in simulation; remove for synth.
  expectedStride :: Unsigned 32
  expectedStride = fromIntegral (rowStrideBytesI8E @ModelDimension)

  prevCapIdx  = register maxBound $ mux dvRise capturedRowReq prevCapIdx  -- Use maxBound as sentinel
  prevCapAddr = register 0 $ mux dvRise capturedAddr  prevCapAddr

  dramRowOutLive = assertStrideOnCommit dvRise prevCapIdx capturedRowReq 
    prevCapAddr capturedAddr expectedStride dramRowAfterEqCheck
    where
      -- Only check stride if previous row was actually committed (not initial sentinel)
      assertStrideOnCommit commitEdge prevIdx curIdx prevAddr curAddr strideB rowSig =
        mux commitEdge (check <$> prevIdx <*> curIdx <*> prevAddr <*> curAddr <*> rowSig) rowSig
        where
          check p c pa ca r =
            let pI = fromEnum p :: Int
                cI = fromEnum c :: Int
                -- Skip check if this is the first real commit (prevIdx == maxBound)
                firstCommit = (p == maxBound)
                sequential = (cI == pI + 1)
                ok = firstCommit || not sequential || (ca == pa + strideB)
            in if ok then r
              else P.error $ "Row stride mismatch at commit: prevIdx=" P.++ show pI
                          P.++ " currIdx=" P.++ show cI
                          P.++ " prevAddr=" P.++ show pa
                          P.++ " currAddr=" P.++ show ca
                          P.++ " expected stride=" P.++ show strideB

  loaderOutput :: WeightLoaderOutput dom
  loaderOutput = WeightLoaderOutput
    { hcRowOut          = hcRowCommitted
    , dramRowOut        = dramRowOutLive
    , dbgRequestedAddr  = liveRowAddr
    , dbgLoadState      = loadState
    , dbgFetchTrigger   = fetchTrigger
    , dbgMultiWordValid = fetchValid
    , dbgFetchedWords   = fetchedWords
    , dbgCapturedRowReq = capturedRowReq
    , dbgCapturedAddr   = capturedAddr
    }

-- Update the assertion function signature:
assertRowsMatchOnCommit :: forall dom n. (KnownNat n, KnownNat (If (OrdCond (CmpNat n 63) 'True 'True 'False) 1 (1 + Div n 64) T.* 64))
  => Signal dom Bool                   -- ^ commit edge (dvRise)
  -> Signal dom (Index HeadDimension)  -- ^ captured row index
  -> Signal dom (Unsigned 32)          -- ^ captured address
  -> Signal dom (RowI8E n)             -- ^ committed DRAM row
  -> Signal dom (RowI8E n)             -- ^ committed HC row
  -> Signal dom (Vec (STREAM.WordsPerRow n) (BitVector 512))  -- ^ raw fetched words
  -> Signal dom (RowI8E n)
assertRowsMatchOnCommit commitEdge capIdx capAddr dramRow hcRow fetchedWords =
  mux commitEdge (check <$> capIdx <*> capAddr <*> dramRow <*> hcRow <*> fetchedWords) dramRow
 where
  check :: Index HeadDimension -> Unsigned 32 -> RowI8E n -> RowI8E n
        -> Vec (STREAM.WordsPerRow n) (BitVector 512) -> RowI8E n
  check ri ad dr hr words' =
    let de = rowExponent dr
        he = rowExponent hr
        dm = rowMantissas dr
        hm = rowMantissas hr
        expMatch  = de == he
        mantMatch = dm == hm
        nShow = 8 :: Int
        showPrefix xs = P.take nShow (P.map show (toList xs))

        -- Parse the raw words to see what was actually fetched
        parsedFromWords :: RowI8E n
        parsedFromWords = STREAM.multiWordRowParser words'
        pw_exp = rowExponent parsedFromWords
        pw_mants = rowMantissas parsedFromWords
    in if expMatch && mantMatch
         then dr
         else
           let dm0 = showPrefix dm
               hm0 = showPrefix hm
               pw0 = showPrefix pw_mants
               -- Show first word in hex for detailed inspection
               word0_hex = case toList words' of
                             (w:_) -> show w
                             _     -> "no words"
           in P.error $
                "DRAM/HC mismatch at commit! row=" P.++ show ri
             P.++ " addr=" P.++ show ad
             P.++ " expMatch=" P.++ show expMatch
             P.++ " mantMatch=" P.++ show mantMatch
             P.++ " dramExp=" P.++ show de
             P.++ " hcExp="   P.++ show he
             P.++ "\n  DRAM mant[0..7]=" P.++ show dm0
             P.++ "\n  HC   mant[0..7]=" P.++ show hm0
             P.++ "\n  Parsed from fetchedWords exp=" P.++ show pw_exp
             P.++ "\n  Parsed from fetchedWords mant[0..7]=" P.++ show pw0
             P.++ "\n  Raw word[0] (hex)=" P.++ word0_hex
             P.++ "\n  (Assertion: DRAM row should equal HC row; fetchedWords shown for debugging)"
