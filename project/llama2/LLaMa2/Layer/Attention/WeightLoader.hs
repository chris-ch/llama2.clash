module LLaMa2.Layer.Attention.WeightLoader
  ( weightLoader, WeightLoaderOutput(..)
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightStreaming as STREAM
import qualified Simulation.Parameters as PARAM
import qualified Prelude as P

data LoadState = LIdle | LFetching | LDone
    deriving (Show, Eq, Generic, NFDataX)

data WeightLoaderOutput dom = WeightLoaderOutput {
  hcRowOut :: Signal dom (RowI8E ModelDimension)
  , dramRowOut :: Signal dom (RowI8E ModelDimension)
  -- Debug additions:
  , dbgRequestedAddr :: Signal dom (Unsigned 32)
  , dbgLoadState :: Signal dom LoadState
  , dbgFetchTrigger :: Signal dom Bool
  , dbgMultiWordValid :: Signal dom Bool  -- The raw fetchValid from multi-word fetcher
}

weightLoader :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool                    -- NEW: dataConsumed (rowDone from processor)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool
     , Signal dom Bool)
weightLoader dramSlaveIn layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed params =
  (axiMaster, loaderOutput, dramDataValid, dramReady)
  where
    -- ==== Hardcoded path ====
    hcWeights :: MatI8E HeadDimension ModelDimension
    hcWeights =
      PARAM.wqHeadQ
        (PARAM.headsQ (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)

    -- Capture rowReq when fetch is triggered
    capturedRowReq :: Signal dom (Index HeadDimension)
    capturedRowReq = register 0 $ mux fetchTrigger rowReq capturedRowReq

    hcRow :: Signal dom (RowI8E ModelDimension)
    hcRow = (!!) hcWeights <$> capturedRowReq

    -- ==== DRAM path ====
    rowAddr :: Signal dom (Unsigned 32)
    rowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowReq

    loadState :: Signal dom LoadState
    loadState = register LIdle nextState
      where
        nextState = withLoad <$> loadState <*> rowReqValid <*> fetchValid <*> downstreamReady <*> dataConsumed
        withLoad LIdle     rv _  _  _  = if rv then LFetching else LIdle
        withLoad LFetching _  fv _  _  = if fv then LDone     else LFetching
        -- KEY FIX: Only transition to LIdle when BOTH downstream is ready AND processor consumed the data
        withLoad LDone     _  _  rd dc = if rd && dc then LIdle else LDone

    dramReady :: Signal dom Bool
    dramReady = loadState .==. pure LIdle

    dramDataValid :: Signal dom Bool
    dramDataValid = loadState .==. pure LDone

    -- Reset-safe trigger
    outOfReset     = register False (pure True)
    outOfResetPrev = register False outOfReset
    reqDuringReset = register False (rowReqValid .&&. dramReady)
    replayFirst    = (not <$> outOfResetPrev) .&&. outOfReset .&&. reqDuringReset

    fetchTrigger :: Signal dom Bool
    fetchTrigger = (rowReqValid .&&. dramReady .&&. outOfReset) .||. replayFirst

    -- AXI burst fetch
    (axiMaster, fetchedWords, fetchValid, _requestReady) =
      STREAM.axiMultiWordRowFetcher @_ @ModelDimension dramSlaveIn fetchTrigger rowAddr

    parsedRow :: Signal dom (RowI8E ModelDimension)
    parsedRow = STREAM.multiWordRowParser <$> fetchedWords

    zeroRow :: RowI8E ModelDimension
    zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

    dramRowRaw :: Signal dom (RowI8E ModelDimension)
    dramRowRaw = regEn zeroRow fetchValid parsedRow
    
    hcRowLatched :: Signal dom (RowI8E ModelDimension)
    hcRowLatched = regEn zeroRow fetchValid hcRow
    
    wasLDone :: Signal dom Bool
    wasLDone = register False dramDataValid
    
    dataJustLatched :: Signal dom Bool
    dataJustLatched = dramDataValid .&&. (not <$> wasLDone)
    
    dramRow :: Signal dom (RowI8E ModelDimension)
    dramRow = assertRowsMatchLazy dataJustLatched capturedRowReq rowAddr dramRowRaw hcRowLatched

    loaderOutput = WeightLoaderOutput 
      { hcRowOut = hcRowLatched
      , dramRowOut = dramRow
      , dbgRequestedAddr = rowAddr
      , dbgLoadState = loadState
      , dbgFetchTrigger = fetchTrigger
      , dbgMultiWordValid = fetchValid
      }

-- | Lazy assertion that only evaluates the comparison when guard is True.
assertRowsMatchLazy :: forall dom n. KnownNat n 
  => Signal dom Bool              -- ^ guard: only check when True
  -> Signal dom (Index HeadDimension)  -- ^ row index (for error message)
  -> Signal dom (Unsigned 32)     -- ^ address (for error message)
  -> Signal dom (RowI8E n)        -- ^ DRAM row
  -> Signal dom (RowI8E n)        -- ^ Hardcoded row
  -> Signal dom (RowI8E n)
assertRowsMatchLazy guard rowIdx addr dramRow hcRow = result
  where
    result = mux guard
                 (checkRow <$> rowIdx <*> addr <*> dramRow <*> hcRow)
                 dramRow

    checkRow :: Index HeadDimension -> Unsigned 32 -> RowI8E n -> RowI8E n -> RowI8E n
    checkRow ri ad dr hr =
      let dramExp   = rowExponent dr
          hcExp     = rowExponent hr
          dramMants = rowMantissas dr
          hcMants   = rowMantissas hr
          expMatch  = dramExp == hcExp
          mantMatch = dramMants == hcMants  -- Check ALL mantissas
      in if expMatch && mantMatch
           then dr
           else errorX $ "DRAM/HC mismatch! row=" P.++ show ri 
                      P.++ " addr=" P.++ show ad
                      P.++ " expMatch=" P.++ show expMatch
                      P.++ " mantMatch=" P.++ show mantMatch
                      P.++ " dramExp=" P.++ show dramExp 
                      P.++ " hcExp=" P.++ show hcExp


-- | Enhanced assertion with more debug info
assertRowsMatchLazyDebug :: forall dom n. KnownNat n 
  => Signal dom Bool                   -- ^ guard
  -> Signal dom (Index HeadDimension)  -- ^ captured row index
  -> Signal dom (Unsigned 32)          -- ^ captured address
  -> Signal dom (Index HeadDimension)  -- ^ current row index (for comparison)
  -> Signal dom (RowI8E n)             -- ^ DRAM row
  -> Signal dom (RowI8E n)             -- ^ HC row
  -> Signal dom (RowI8E n)
assertRowsMatchLazyDebug guard capturedIdx capturedAddr currentIdx dramRow hcRow = result
  where
    result = mux guard
                 (checkRow <$> capturedIdx <*> capturedAddr <*> currentIdx <*> dramRow <*> hcRow)
                 dramRow

    checkRow :: Index HeadDimension -> Unsigned 32 -> Index HeadDimension 
             -> RowI8E n -> RowI8E n -> RowI8E n
    checkRow capIdx capAddr curIdx dr hr =
      let dramExp   = rowExponent dr
          hcExp     = rowExponent hr
          dramMants = rowMantissas dr
          hcMants   = rowMantissas hr
          expMatch  = dramExp == hcExp
          mantMatch = dramMants == hcMants
      in if expMatch && mantMatch
           then dr
           else errorX $ "DRAM/HC mismatch! capturedRow=" P.++ show capIdx 
                      P.++ " currentRow=" P.++ show curIdx
                      P.++ " capturedAddr=" P.++ show capAddr
                      P.++ " expMatch=" P.++ show expMatch
                      P.++ " mantMatch=" P.++ show mantMatch
                      P.++ " dramExp=" P.++ show dramExp 
                      P.++ " hcExp=" P.++ show hcExp
                      P.++ " dramMant0=" P.++ show (P.head (toList dramMants))
                      P.++ " hcMant0=" P.++ show (P.head (toList hcMants))
