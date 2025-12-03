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
import qualified LLaMa2.Memory.WeightsLayout as Layout
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
  , dbgRequestedAddr  :: Signal dom (Unsigned 32)
  , dbgLoadState      :: Signal dom LoadState
  , dbgFetchTrigger   :: Signal dom Bool
  , dbgMultiWordValid :: Signal dom Bool
  , dbgFetchedWords   :: Signal dom (Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
  , dbgCapturedRowReq :: Signal dom (Index HeadDimension)
  , dbgCapturedAddr   :: Signal dom (Unsigned 32)
  , dbgRowReqEffective    :: Signal dom (Index HeadDimension)
  , dbgReplayFirst        :: Signal dom Bool
  , dbgExpectedAddrFromRow :: Signal dom (Unsigned 32)
  , dbgAddrMismatch       :: Signal dom Bool
  , dbgHcRowFromCapture   :: Signal dom (RowI8E ModelDimension)
  }

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
weightLoader dramSlaveIn layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed params =
  (axiMaster, loaderOutput, weightValidLevel, weightReadyLevel)
 where
  hcWeights :: MatI8E HeadDimension ModelDimension
  hcWeights =
    PARAM.qMatrix
      (PARAM.qHeads (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)

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

  prevValid = register False weightValidLevel
  dvRise    = weightValidLevel .&&. (not <$> prevValid)

  outOfReset     = register False (pure True)
  outOfResetPrev = register False outOfReset

  reqDuringResetFlag :: Signal dom Bool
  reqDuringResetFlag = register False (rowReqValid .&&. weightReadyLevel)

  reqDuringResetRow :: Signal dom (Index HeadDimension)
  reqDuringResetRow = regEn 0 (rowReqValid .&&. weightReadyLevel) rowReq

  replayFirst :: Signal dom Bool
  replayFirst = (not <$> outOfResetPrev) .&&. outOfReset .&&. reqDuringResetFlag

  fetchTrigger :: Signal dom Bool
  fetchTrigger =
    (rowReqValid .&&. weightReadyLevel .&&. outOfReset) .||. replayFirst

  rowReqEffective :: Signal dom (Index HeadDimension)
  rowReqEffective = mux replayFirst reqDuringResetRow rowReq

  liveRowAddr :: Signal dom (Unsigned 32)
  liveRowAddr = Layout.rowAddressCalculator Layout.QMatrix layerIdx headIdx <$> rowReqEffective

  capturedAddr :: Signal dom (Unsigned 32)
  capturedAddr = register 0 $ mux fetchTrigger liveRowAddr capturedAddr

  (axiMaster, fetchedWords, fetchValid, _requestReady) =
      Layout.axiMultiWordRowFetcher @_ @ModelDimension dramSlaveIn fetchTrigger liveRowAddr

  parsedRow :: Signal dom (RowI8E ModelDimension)
  parsedRow = Layout.multiWordRowParser <$> fetchedWords

  zeroRow :: RowI8E ModelDimension
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  dramRowAssembled :: Signal dom (RowI8E ModelDimension)
  dramRowAssembled = regEn zeroRow fetchValid parsedRow

  dramRowCommitted :: Signal dom (RowI8E ModelDimension)
  dramRowCommitted = regEn zeroRow dvRise dramRowAssembled

  capturedRowReq :: Signal dom (Index HeadDimension)
  capturedRowReq = register 0 $ mux fetchTrigger rowReqEffective capturedRowReq

  hcRowAtAssemble :: Signal dom (RowI8E ModelDimension)
  hcRowAtAssemble = regEn zeroRow fetchValid ((!!) hcWeights <$> capturedRowReq)

  hcRowCommitted :: Signal dom (RowI8E ModelDimension)
  hcRowCommitted = regEn zeroRow dvRise hcRowAtAssemble

  fetchedWordsAssembled :: Signal dom (Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsAssembled = regEn (repeat 0) fetchValid fetchedWords

  fetchedWordsCommitted :: Signal dom (Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsCommitted = regEn (repeat 0) dvRise fetchedWordsAssembled

  expectedStride :: Unsigned 32
  expectedStride = fromIntegral (Layout.rowStrideBytesI8E @ModelDimension)

  prevCapIdx  = register maxBound $ mux dvRise capturedRowReq prevCapIdx
  prevCapAddr = register 0 $ mux dvRise capturedAddr prevCapAddr

  expectedAddrFromRow :: Signal dom (Unsigned 32)
  expectedAddrFromRow = Layout.rowAddressCalculator Layout.QMatrix layerIdx headIdx <$> capturedRowReq

  addrMismatch :: Signal dom Bool
  addrMismatch = (/=) <$> capturedAddr <*> expectedAddrFromRow

  hcRowFromCapture :: Signal dom (RowI8E ModelDimension)
  hcRowFromCapture = regEn zeroRow fetchTrigger ((!!) hcWeights <$> rowReqEffective)

  -- FIXED: Added hcRowFromCapture as 7th argument
  dramRowAfterEqCheck = assertRowsMatchOnCommitEnhanced 
    dvRise 
    capturedRowReq 
    capturedAddr
    expectedAddrFromRow 
    dramRowCommitted 
    hcRowCommitted 
    hcRowFromCapture        -- This was missing!
    fetchedWordsCommitted 
    layerIdx 
    headIdx

  dramRowOutLive = assertStrideOnCommit dvRise prevCapIdx capturedRowReq 
    prevCapAddr capturedAddr expectedStride dramRowAfterEqCheck
    where
      assertStrideOnCommit commitEdge prevIdx curIdx prevAddr curAddr strideB rowSig =
        mux commitEdge (check <$> prevIdx <*> curIdx <*> prevAddr <*> curAddr <*> rowSig) rowSig
        where
          check p c pa ca r =
            let pI = fromEnum p :: Int
                cI = fromEnum c :: Int
                firstCommit = (p == maxBound)
                sequential = (cI == pI + 1)
                ok = firstCommit || not sequential || (ca == pa + strideB)
            in if ok then r
              else P.error $ "Row stride mismatch at commit: prevIdx=" P.++ show pI
                          P.++ " currIdx=" P.++ show cI
                          P.++ " prevAddr=" P.++ show pa
                          P.++ " currAddr=" P.++ show ca
                          P.++ " expected stride=" P.++ show strideB

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
    , dbgRowReqEffective    = rowReqEffective
    , dbgReplayFirst        = replayFirst
    , dbgExpectedAddrFromRow = expectedAddrFromRow
    , dbgAddrMismatch       = addrMismatch
    , dbgHcRowFromCapture   = hcRowFromCapture
    }

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
