module LLaMa2.Layer.FeedForward.FeedForwardNetwork (
   feedForwardStage
) where

import Clash.Prelude
import LLaMa2.Numeric.RmsNormSeq (rmsNormSeq)
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

    -- Sequential rmsNorm: triggered on the rising edge of rmsFfnValid.
    rmsFfnDone = rmsFfnValid .&&. (not <$> register False rmsFfnValid)
    (rmsNormValid, xHat, _) = rmsNormSeq rmsFfnDone inputVector rmsFfnVec

    -- effectiveValidIn: fires once on the rising edge of rmsNormValid while pending.
    -- pendingInput clears on effectiveValidIn (not on rmsNormValid directly —
    -- rmsNormValid from the previous layer is still True when the next validInRise
    -- fires, which would prematurely clear pendingInput before the new run starts).
    effectiveValidIn = pendingInput .&&. rmsNormValid .&&. (not <$> register False rmsNormValid)

    pendingInput = register False nextPendingInput
     where
      nextPendingInput =
        mux effectiveValidIn (pure False) $
        mux validInRise (pure True)
        pendingInput

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
