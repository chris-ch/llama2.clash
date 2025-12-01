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
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool
     , Signal dom Bool)
weightLoader dramSlaveIn layerIdx headIdx rowReq rowReqValid downstreamReady params =
  (axiMaster, loaderOutput, dramDataValid, dramReady)
  where
    -- ==== Hardcoded path ====
    hcWeights :: MatI8E HeadDimension ModelDimension
    hcWeights =
      PARAM.wqHeadQ
        (PARAM.headsQ (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)

    hcRow :: Signal dom (RowI8E ModelDimension)
    hcRow = (!!) hcWeights <$> rowReq

    -- ==== DRAM path (using multi-word fetcher) ====
    rowAddr :: Signal dom (Unsigned 32)
    rowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowReq

    loadState :: Signal dom LoadState
    loadState = register LIdle nextState
      where
        nextState = withLoad <$> loadState <*> rowReqValid <*> fetchValid <*> downstreamReady
        withLoad LIdle     rv _  _  = if rv then LFetching else LIdle
        withLoad LFetching _  fv _  = if fv then LDone     else LFetching
        withLoad LDone     _  _  rd = if rd then LIdle     else LDone

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

    -- AXI burst fetch for the whole row
    (axiMaster, fetchedWords, fetchValid, _requestReady) =
      STREAM.axiMultiWordRowFetcher @_ @ModelDimension dramSlaveIn fetchTrigger rowAddr

    -- Parse multi-word response
    parsedRow :: Signal dom (RowI8E ModelDimension)
    parsedRow = STREAM.multiWordRowParser <$> fetchedWords

    -- DEBUG: Print what we're getting before the assertion
    -- This version delays the check by 1 cycle to ensure data is stable
    checkedParsedRow :: Signal dom (RowI8E ModelDimension)
    checkedParsedRow = assertRowsMatchDebug fetchValid rowReq rowAddr parsedRow hcRow

    zeroRow :: RowI8E ModelDimension
    zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

    -- Publish the row (sticky until next fetch)
    dramRow :: Signal dom (RowI8E ModelDimension)
    dramRow = regEn zeroRow fetchValid checkedParsedRow

    loaderOutput = WeightLoaderOutput 
      { hcRowOut = hcRow
      , dramRowOut = dramRow
      , dbgRequestedAddr = rowAddr
      , dbgLoadState = loadState
      , dbgFetchTrigger = fetchTrigger
      , dbgMultiWordValid = fetchValid
      }

-- Enhanced assertion with more debug info
assertRowsMatchDebug :: forall dom n . KnownNat n => Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> Signal dom (Unsigned 32)
  -> Signal dom (RowI8E n)     -- dramRow
  -> Signal dom (RowI8E n)     -- hcRow  
  -> Signal dom (RowI8E n)
assertRowsMatchDebug guard rowIdx addr dramRow hcRow = result
  where
    dramExp = rowExponent <$> dramRow
    hcExp   = rowExponent <$> hcRow
    dramMant0 = (!! (0 :: Int)) . rowMantissas <$> dramRow
    hcMant0 = (!! (0 :: Int)) . rowMantissas <$> hcRow

    expMatch = dramExp .==. hcExp

    -- Create detailed error message
    result = mux (guard .&&. (not <$> expMatch))
                 (mkError <$> rowIdx <*> addr <*> dramExp <*> hcExp <*> dramMant0 <*> hcMant0)
                 dramRow

    mkError ri ad de he dm hm = 
      errorX $ "DRAM/HC mismatch! row=" P.++ show ri 
            P.++ " addr=" P.++ show ad
            P.++ " dramExp=" P.++ show de 
            P.++ " hcExp=" P.++ show he
            P.++ " dramMant[0]=" P.++ show dm
            P.++ " hcMant[0]=" P.++ show hm
