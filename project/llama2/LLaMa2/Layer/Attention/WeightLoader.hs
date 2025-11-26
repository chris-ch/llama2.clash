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
  -> Signal dom (Index HeadDimension)           -- ^ row request
  -> Signal dom Bool                            -- ^ row request valid
  -> Signal dom Bool                            -- ^ downstream ready
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom
     , Signal dom Bool                          -- ^ dramDataValid
     , Signal dom Bool)                         -- ^ dramReady
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

  -- ==== DRAM path ====
  rowAddr :: Signal dom (Unsigned 32)
  rowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowReq

  -- Loader state: single outstanding request
  loadState :: Signal dom LoadState
  loadState = register LIdle nextState
   where
    nextState :: Signal dom LoadState
    nextState = withLoad <$> loadState <*> rowReqValid <*> fetchValid <*> downstreamReady
     where
      withLoad LIdle     rv _  _  = if rv then LFetching else LIdle
      withLoad LFetching _  fv _  = if fv then LDone     else LFetching
      withLoad LDone     _  _  rd = if rd then LIdle     else LDone

  dramReady     :: Signal dom Bool
  dramReady     = loadState .==. pure LIdle

  dramDataValid :: Signal dom Bool
  dramDataValid = loadState .==. pure LDone

  -- Reset-safe fetch trigger:
  -- - outOfReset rises at the first cycle after reset.
  -- - If a request was asserted while in reset (cycle 0), replay it once at cycle 1.
  outOfReset     :: Signal dom Bool
  outOfReset      = register False (pure True)
  
  outOfResetPrev :: Signal dom Bool
  outOfResetPrev  = register False outOfReset

  -- Capture whether a request was present during reset (evaluated in cycle 0).
  reqDuringReset :: Signal dom Bool
  reqDuringReset  = register False (rowReqValid .&&. dramReady)

  replayFirst :: Signal dom Bool
  replayFirst = (not <$> outOfResetPrev) .&&. outOfReset .&&. reqDuringReset

  -- Normal trigger once out of reset; includes replay at the first post-reset cycle.
  fetchTrigger :: Signal dom Bool
  fetchTrigger =
       (rowReqValid .&&. dramReady .&&. outOfReset)
    .||. replayFirst

  -- AXI fetcher: captures fetchTrigger pulses and issues one-beat reads
  (axiMaster, fetchedWord, fetchValid, _requestReady) =
    STREAM.axiRowFetcher dramSlaveIn fetchTrigger rowAddr

  -- Parse and register fetched row
  zeroRow :: RowI8E ModelDimension
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  parsedRow :: Signal dom (RowI8E ModelDimension)
  parsedRow = STREAM.rowParser <$> fetchedWord

  dramRow :: Signal dom (RowI8E ModelDimension)
  dramRow = regEn zeroRow fetchValid parsedRow

