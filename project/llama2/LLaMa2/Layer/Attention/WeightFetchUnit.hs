module LLaMa2.Layer.Attention.WeightFetchUnit
  ( WeightFetchOut(..), WeightFetchIn(..)
  , weightFetchUnit
  ) where

import Clash.Prelude
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Types.ModelConfig
import qualified Simulation.Parameters as PARAM
import qualified Prelude as P
import Clash.Debug (trace)

-- | Trace INPUT_VALID signal
traceInputValidSignal :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom Bool -> Signal dom Bool -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
traceInputValidSignal layerIdx headIdx sig inputValid weightValid ri = traced
  where
    traced = go <$> sig <*> inputValid <*> weightValid <*> ri
    go req iv wv ridx
      | iv        = trace (prefix P.++ "INPUT_VALID wv=" P.++ show wv P.++ " ri=" P.++ show ridx) req
      | otherwise = req
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

--------------------------------------------------------------------------------
-- BLOCK: WeightMismatchTracer
-- Traces when DRAM and HC weights differ
--------------------------------------------------------------------------------
weightMismatchTracer :: Index NumLayers
  -> Signal dom Bool
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom (RowI8E ModelDimension)
weightMismatchTracer layerIdx valid dram hc = result
  where
    result = check <$> valid <*> dram <*> hc
    check v d h
      | v && (rowExponent d P./= rowExponent h P.|| rowMantissas d P./= rowMantissas h) =
          trace ("L" P.++ show layerIdx P.++ " WEIGHT_MISMATCH exp_d=" P.++ show (rowExponent d)
                P.++ " exp_h=" P.++ show (rowExponent h)) d
      | otherwise = d

--------------------------------------------------------------------------------
-- COMPONENT 3: WeightFetchUnit
-- Coordinates weight loading from DRAM via WeightLoader
--------------------------------------------------------------------------------
data WeightFetchIn dom = WeightFetchIn
  { wfRowIndex      :: Signal dom (Index HeadDimension)
  , wfRowReqValid   :: Signal dom Bool  -- Request from compute unit
  , wfConsumeSignal :: Signal dom Bool
  , wfRowDone       :: Signal dom Bool
  } deriving (Generic)

data WeightFetchOut dom = WeightFetchOut
  { wfAxiMaster    :: Master.AxiMasterOut dom
  , wfWeightDram   :: Signal dom (RowI8E ModelDimension)  -- DRAM weights
  , wfWeightHC     :: Signal dom (RowI8E ModelDimension)  -- HC reference weights
  , wfWeightValid  :: Signal dom Bool
  , wfWeightReady  :: Signal dom Bool
  } deriving (Generic)

weightFetchUnit :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> PARAM.DecoderParameters
  -> Signal dom Bool  -- inputValid for tracing
  -> WeightFetchIn dom
  -> WeightFetchOut dom
weightFetchUnit dramSlaveIn layerIdx headIdx params inputValid inputs =
  WeightFetchOut
    { wfAxiMaster    = axiMaster
    , wfWeightDram   = currentRowDramTraced
    , wfWeightHC     = currentRowHC
    , wfWeightValid  = weightValid
    , wfWeightReady  = weightReady
    }
  where
    -- Weight loader integration
    (axiMaster, weightLoaderOut, weightValid, weightReady) =
      LOADER.weightLoader dramSlaveIn layerIdx headIdx
                          (wfRowIndex inputs) rowReqValidTraced 
                          (wfConsumeSignal inputs) (wfRowDone inputs)
                          params

    currentRowDramRaw = LOADER.dramRowOut weightLoaderOut
    currentRowHCRaw   = LOADER.hcRowOut weightLoaderOut

    -- Ensure rows don't change while valid
    currentRowDram = LOADER.assertRowStable weightValid currentRowDramRaw
    currentRowHC   = LOADER.assertRowStable weightValid currentRowHCRaw

    currentRowDramTraced = weightMismatchTracer layerIdx weightValid 
                             currentRowDram currentRowHC

    rowReqValidTraced = traceInputValidSignal layerIdx headIdx 
                          (wfRowReqValid inputs) inputValid weightValid 
                          (wfRowIndex inputs)
