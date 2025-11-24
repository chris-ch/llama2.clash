module LLaMa2.Layer.Attention.WeightLoader
  ( WeightLoaderOut(..)
  , weightLoader
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightStreaming as STREAM
import qualified Simulation.Parameters as PARAM

data WeightLoaderOut dom = WeightLoaderOut
  { wlRowDataDRAM      :: Signal dom (RowI8E ModelDimension)
  , wlRowDataHC        :: Signal dom (RowI8E ModelDimension)
  , wlDRAMValid        :: Signal dom Bool
  , wlHCValid          :: Signal dom Bool
  , wlDRAMReady        :: Signal dom Bool
  }

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
  -> (Master.AxiMasterOut dom, WeightLoaderOut dom)
weightLoader dramSlaveIn layerIdx headIdx rowReq rowReqValid downstreamReady params =
  (axiMaster, WeightLoaderOut dramRow hcRow dramDataValid hcDataValid dramReady)
 where
  -- ==== Hardcoded path ====
  hcWeights = PARAM.wqHeadQ (PARAM.headsQ (PARAM.multiHeadAttention (PARAM.modelLayers params !! layerIdx)) !! headIdx)
  hcRow = (!!) hcWeights <$> rowReq
  hcDataValid = rowReqValid
  
  -- ==== DRAM path ====
  rowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowReq
  fetchTrigger = rowReqValid .&&. register True dramReady
  
  (axiMaster, fetchedWord, fetchValid) = 
    STREAM.axiRowFetcher dramSlaveIn fetchTrigger rowAddr
  
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }
  parsedRow = STREAM.parseRow <$> fetchedWord
  
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
