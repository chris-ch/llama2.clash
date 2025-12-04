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
  , dbgCapturedAddr   :: Signal dom (Unsigned 32)
  , dbgCapturedRowReq :: Signal dom (Index HeadDimension)
  , dbgLoadState      :: Signal dom LoadState
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

  weightReady :: Signal dom Bool
  weightReady = loadState .==. pure LIdle

  weightValid :: Signal dom Bool
  weightValid = loadState .==. pure LDone

  -- Rising edge when a new row becomes valid to the downstream
  prevValid = register False weightValid
  dvRise    = weightValid .&&. (not <$> prevValid)

  -- Live request and address (combinational)
  liveRow  :: Signal dom (Index HeadDimension)
  liveRow  = rowReq

  liveAddr :: Signal dom (Unsigned 32)
  liveAddr = Layout.rowAddressCalculator Layout.QMatrix layerIdx headIdx <$> liveRow

  -- Start a new fetch only when idle and the request pulses
  fetchTrigger :: Signal dom Bool
  fetchTrigger = weightReady .&&. rowReqValid

  -- Transaction record captured exactly at the same handshake that starts the AXI read
  -- NOTE: We capture (row, addr) on fetchTrigger. The strict fetcher also latches liveAddr
  -- on this same pulse, removing any possibility of skew.
  txnReg :: Signal dom Txn
  txnReg = regEn (Txn 0 0) fetchTrigger (Txn <$> liveRow <*> liveAddr)

  -- AXI multi-word fetcher (STRICT!):
  -- Feed it liveAddr so that both it and txnReg “see” the exact same value at fetchTrigger.
  (axiMaster, fetchedWords, fetchValid, _readyStrict) =
    Layout.axiMultiWordRowFetcher @_ @ModelDimension dram fetchTrigger liveAddr

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

  -- Keep the words for diagnostics (parsed back in the assertion)
  fetchedWordsAssembled
    :: Signal dom (Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsAssembled = regEn (repeat 0) fetchValid fetchedWords

  fetchedWordsCommitted
    :: Signal dom (Vec (Layout.WordsPerRow ModelDimension) (BitVector 512))
  fetchedWordsCommitted = regEn (repeat 0) dvRise fetchedWordsAssembled

  -- HC rows aligned with commit
  hcRowCommitted :: Signal dom (RowI8E ModelDimension)
  hcRowCommitted = regEn zeroRow dvRise ((!!) hcWeights . tRow <$> txnReg)

  -- HC row captured at the same time as the fetch handshake (for assert text)
  hcRowFromCapture :: Signal dom (RowI8E ModelDimension)
  hcRowFromCapture = regEn zeroRow fetchTrigger ((!!) hcWeights <$> liveRow)

  -- Expected address recomputed from the captured row index
  expectedAddrFromRow :: Signal dom (Unsigned 32)
  expectedAddrFromRow =
    Layout.rowAddressCalculator Layout.QMatrix layerIdx headIdx . tRow <$> txnReg

  -- COMMIT-TIME CHECK: do not remove or downgrade.
  dramRowAfterEqCheck :: Signal dom (RowI8E ModelDimension)
  dramRowAfterEqCheck =
    assertRowsMatchOnCommitEnhanced
      dvRise
      (tRow  <$> txnReg)              -- capIdx
      (tAddr <$> txnReg)              -- capAddr
      expectedAddrFromRow             -- expectedAddr
      dramRowCommitted                -- dramRow
      hcRowCommitted                  -- hcRow
      hcRowFromCapture               -- hcRowCapture
      fetchedWordsCommitted           -- fetchedWords'
      layerIdx
      headIdx

  -- Loader FSM next-state
  nextState =
    mux (loadState .==. pure LIdle .&&. rowReqValid)
        (pure LFetching)
    $ mux (loadState .==. pure LFetching .&&. fetchValid)
        (pure LDone)
    $ mux (loadState .==. pure LDone .&&. downstreamReady .&&. dataConsumed)
        (pure LIdle)
        loadState

  -- Outputs (dramRowOut is post-assertion)
  out :: WeightLoaderOutput dom
  out = WeightLoaderOutput
    { hcRowOut          = hcRowCommitted
    , dramRowOut        = dramRowAfterEqCheck
    , dbgRequestedAddr  = liveAddr
    , dbgCapturedAddr   = tAddr <$> txnReg
    , dbgCapturedRowReq = tRow  <$> txnReg
    , dbgLoadState      = loadState
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

data Txn = Txn
  { tRow  :: Index HeadDimension
  , tAddr :: Unsigned 32
  } deriving (Generic, NFDataX, Show, Eq)
