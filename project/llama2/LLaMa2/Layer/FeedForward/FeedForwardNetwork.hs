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
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified LLaMa2.Memory.FPVecLoader as FPVec
feedForwardStage
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                      -- ^ cycle counter
  -> Slave.AxiSlaveIn dom                          -- ^ DRAM slave
  -> Signal dom (Index NumLayers)                  -- ^ layer index
  -> Signal dom Bool                               -- ^ validIn
  -> Signal dom Bool                               -- ^ readyIn (from downstream)
  -> Signal dom (Vec ModelDimension FixedPoint)    -- ^ input vector
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec ModelDimension FixedPoint)  -- ^ output vector (residual added)
     , Signal dom Bool                             -- ^ validOut
     , Signal dom Bool                             -- ^ readyOut (to upstream)
     )
feedForwardStage cycleCounter dramSlaveIn layerIdx validIn readyIn inputVector =
  (axiMasterOut, outputVector, validOut, readyOut)
  where
    --------------------------------------------------------------------------
    -- DRAM-backed fRMSFfnF fetch (once per inputValid; gates FFN start)
    --------------------------------------------------------------------------
    -- Rising edge: fpVecLoader needs a one-shot trigger (processingControllerFSM
    -- keeps 'enable' high for the entire PROCESSING_RUN state).
    validInRise = validIn .&&. (not <$> register False validIn)

    (rmsFfnAxiMaster, rmsFfnVec, rmsFfnValid, rmsFfnBusy) =
      FPVec.fpVecLoader cycleCounter dramSlaveIn
        validInRise
        (Layout.rmsFfnAddress <$> layerIdx)

    -- Hold validInRise latch until fRMSFfn fetch completes
    pendingInput = register False nextPendingInput
     where
      nextPendingInput =
        mux (pendingInput .&&. rmsFfnValid) (pure False) $
        mux validInRise (pure True)
        pendingInput

    effectiveValidIn = pendingInput .&&. rmsFfnValid

    -- Pre-normalize with DRAM-fetched RMS weights
    xHat = rmsNormFwFix <$> inputVector <*> rmsFfnVec

    --------------------------------------------------------------------------
    -- DRAM-backed FFN core: W1 (gate) -> W3 (up) -> W2 (down)
    --------------------------------------------------------------------------
    (ffnAxiMaster, ffnCore, coreValidOut, readyOut) =
      ffnProjector cycleCounter dramSlaveIn layerIdx effectiveValidIn readyIn xHat

    -- rmsAtt fetch has priority (finishes before FFN starts due to pendingInput gate)
    axiMasterOut = Master.axiMasterMux rmsFfnBusy rmsFfnAxiMaster ffnAxiMaster

    -- Residual connection: register inputVector aligned with core output timing
    inputVectorDelayed = regEn (repeat 0) coreValidOut inputVector
    outputVector = zipWith (+) <$> inputVectorDelayed <*> ffnCore

    validOut = coreValidOut
