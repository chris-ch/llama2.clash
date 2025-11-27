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

data LoadState = LIdle | LFetching | LDone
    deriving (Show, Eq, Generic, NFDataX)

data WeightLoaderOutput dom = WeightLoaderOutput {
  hcRowOut :: Signal dom (RowI8E ModelDimension)
  , dramRowOut :: Signal dom (RowI8E ModelDimension)
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
  (axiMaster, WeightLoaderOutput {hcRowOut = hcRow, dramRowOut = dramRow}, dramDataValid, dramReady)
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
    
    -- State machine
    loadState :: Signal dom LoadState
    loadState = register LIdle nextState
      where
        nextState = withLoad <$> loadState <*> rowReqValid <*> fetchValid <*> downstreamReady
          where
            withLoad LIdle     rv _  _  = if rv then LFetching else LIdle
            withLoad LFetching _  fv _  = if fv then LDone     else LFetching
            withLoad LDone     _  _  rd = if rd then LIdle     else LDone
    
    dramReady :: Signal dom Bool
    dramReady = loadState .==. pure LIdle
    
    dramDataValid :: Signal dom Bool
    dramDataValid = loadState .==. pure LDone
    
    -- Reset-safe trigger (unchanged)
    outOfReset = register False (pure True)
    outOfResetPrev = register False outOfReset
    reqDuringReset = register False (rowReqValid .&&. dramReady)
    replayFirst = (not <$> outOfResetPrev) .&&. outOfReset .&&. reqDuringReset
    
    fetchTrigger :: Signal dom Bool
    fetchTrigger = (rowReqValid .&&. dramReady .&&. outOfReset) .||. replayFirst
    
    -- Multi-word AXI fetcher
    (axiMaster, fetchedWords, fetchValid, _requestReady) =
      STREAM.axiMultiWordRowFetcher @_ @ModelDimension dramSlaveIn fetchTrigger rowAddr
    
    -- Parse multi-word response
    zeroRow :: RowI8E ModelDimension
    zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }
    
    parsedRow :: Signal dom (RowI8E ModelDimension)
    parsedRow = STREAM.multiWordRowParser <$> fetchedWords
    
    dramRow :: Signal dom (RowI8E ModelDimension)
    dramRow = regEn zeroRow fetchValid parsedRow
