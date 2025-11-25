module LLaMa2.Layer.Attention.WeightLoader
  ( weightLoader
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
  
weightLoader :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)           -- ^ row request
  -> Signal dom Bool                            -- ^ row request valid
  -> Signal dom Bool                            -- ^ downstream ready
  -> PARAM.DecoderParameters
  -> (Master.AxiMasterOut dom, Signal dom (RowI8E ModelDimension), Signal dom Bool, Signal dom Bool)
weightLoader dramSlaveIn layerIdx headIdx rowReq rowReqValid downstreamReady params =
  (axiMaster, hcRow, dramDataValid, dramReady) -- MANUALLY CHANGE HERE: hcRow for constant params, dramRow for DRAM loaded
 where
  -- ==== Hardcoded path ====
  hcWeights :: MatI8E HeadDimension ModelDimension
  hcWeights = PARAM.wqHeadQ (PARAM.headsQ (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)

  hcRow :: Signal dom (RowI8E ModelDimension)
  hcRow = (!!) hcWeights <$> rowReq
  
  -- ==== DRAM path ====
  rowAddr :: Signal dom (Unsigned 32)
  rowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowReq

  fetchTrigger :: Signal dom Bool
  fetchTrigger = rowReqValid .&&. register True dramReady
  
  fetchedWord :: Signal dom (BitVector 512)
  (axiMaster, fetchedWord, fetchValid) = 
    STREAM.axiRowFetcher dramSlaveIn fetchTrigger rowAddr
  
  zeroRow :: RowI8E ModelDimension
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  parsedRow :: Signal dom (RowI8E ModelDimension)
  parsedRow = STREAM.parseRow <$> fetchedWord

  dramRow :: Signal dom (RowI8E ModelDimension)  
  dramRow = register zeroRow nextDramRow
    where nextDramRow = mux fetchValid parsedRow dramRow
  
  loadState :: Signal dom LoadState
  loadState = register LIdle nextState
    where
      nextState = mux (loadState .==. pure LIdle .&&. rowReqValid)
                      (pure LFetching)
                      (mux (loadState .==. pure LFetching .&&. fetchValid)
                           (pure LDone)
                           (mux (loadState .==. pure LDone .&&. downstreamReady)
                                (pure LIdle)
                                loadState))
  
  dramReady = loadState .==. pure LIdle
  dramDataValid = loadState .==. pure LDone
