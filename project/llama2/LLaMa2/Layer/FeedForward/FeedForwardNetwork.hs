module LLaMa2.Layer.FeedForward.FeedForwardNetwork (
   feedForwardStage
) where

import Clash.Prelude
import LLaMa2.Numeric.FixedPoint ( rmsNormFwFix )
import LLaMa2.Types.ModelConfig
    ( ModelDimension, NumLayers )
import LLaMa2.Numeric.Types ( FixedPoint )

import LLaMa2.Layer.FeedForward.FFNProjector (ffnProjector)
import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified Simulation.Parameters as PARAM (FeedForwardNetworkComponentQ (..), DecoderParameters)

feedForwardStage
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                      -- ^ cycle counter
  -> Slave.AxiSlaveIn dom                          -- ^ DRAM slave
  -> Index NumLayers                               -- ^ layer index
  -> Signal dom Bool                               -- ^ validIn
  -> Signal dom Bool                               -- ^ readyIn (from downstream)
  -> PARAM.FeedForwardNetworkComponentQ            -- ^ HC params (for fRMSFfnF)
  -> Signal dom (Vec ModelDimension FixedPoint)    -- ^ input vector
  -> PARAM.DecoderParameters                       -- ^ full params for weight loaders
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec ModelDimension FixedPoint)  -- ^ output vector (residual added)
     , Signal dom Bool                             -- ^ validOut
     , Signal dom Bool                             -- ^ readyOut (to upstream)
     )
feedForwardStage cycleCounter dramSlaveIn layerIdx validIn readyIn ffn inputVector params =
  (ffnAxiMaster, outputVector, validOut, readyOut)
  where
    -- Pre-normalize the input (combinational, stays HC)
    xHat = rmsNormFwFix <$> inputVector <*> pure (PARAM.fRMSFfnF ffn)

    -- DRAM-backed FFN core: W1 (gate) -> W3 (up) -> W2 (down)
    (ffnAxiMaster, ffnCore, coreValidOut, readyOut) =
      ffnProjector cycleCounter dramSlaveIn layerIdx validIn readyIn xHat params

    -- Residual connection: register inputVector aligned with core output timing
    inputVectorDelayed = regEn (repeat 0) coreValidOut inputVector
    outputVector = zipWith (+) <$> inputVectorDelayed <*> ffnCore

    validOut = coreValidOut
