module LLaMa2.Layer.FeedForward.FeedForwardNetwork (
   feedForwardStage
) where

import Clash.Prelude
import qualified GHC.TypeNats as TN
import LLaMa2.Numeric.RmsNormSeq (rmsNormSeq)
import LLaMa2.Types.ModelConfig
    ( ModelDimension, NumLayers )
import LLaMa2.Numeric.Types ( FixedPoint )
import LLaMa2.Types.LayerData (ActivationBramAddr)

import LLaMa2.Layer.FeedForward.FFNProjector (ffnProjector)
import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified LLaMa2.Memory.FPVecLoader as FPVec

-- Base address of slot 2 (attentionOutput) in the activation BRAM.
slot2BramBase :: ActivationBramAddr
slot2BramBase = natToNum @(2 TN.* ModelDimension)

-- Base address of slot 3 (feedForwardOutput) in the activation BRAM.
slot3BramBase :: ActivationBramAddr
slot3BramBase = natToNum @(3 TN.* ModelDimension)

feedForwardStage
  :: HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                               -- ^ cycle counter
  -> Slave.AxiSlaveIn dom                                   -- ^ DRAM slave
  -> Signal dom (Index NumLayers)                           -- ^ layer index
  -> Signal dom Bool                                        -- ^ validIn
  -> Signal dom Bool                                        -- ^ readyIn (from downstream)
  -> Signal dom FixedPoint                                  -- ^ bramRdData (BRAM slot 2, 1-cycle latency)
  -> ( Master.AxiMasterOut dom
     , Signal dom Bool                                      -- ^ writeDone (slot 3 fully written)
     , Signal dom Bool                                      -- ^ readyOut (to upstream)
     , Signal dom ActivationBramAddr                        -- ^ bramRdAddr (drive to BRAM read port)
     , Signal dom (Maybe (ActivationBramAddr, FixedPoint))  -- ^ bramWrite (drive to BRAM write port)
     )
feedForwardStage cycleCounter dramSlaveIn layerIdx validIn readyIn bramRdData =
  (axiMasterOut, writeDone, readyOut, bramRdAddr, bramWrite)
  where
    --------------------------------------------------------------------------
    -- DRAM-backed fRMSFfnF fetch (once per inputValid; gates FFN start)
    --------------------------------------------------------------------------
    validInRise = validIn .&&. (not <$> register False validIn)

    (rmsFfnAxiMaster, rmsFfnVec, rmsFfnValid, rmsFfnBusy) =
      FPVec.fpVecLoader cycleCounter dramSlaveIn
        validInRise
        (Layout.rmsFfnAddress <$> layerIdx)

    -- Sequential rmsNorm: bramRdData supplies xi element-by-element.
    -- rdNext drives the BRAM read address one cycle ahead (1-cycle BRAM latency).
    rmsFfnDone = rmsFfnValid .&&. (not <$> register False rmsFfnValid)
    (rmsNormValid, xHat, _, rdNext) = rmsNormSeq rmsFfnDone bramRdData rmsFfnVec

    effectiveValidIn = pendingInput .&&. rmsNormValid .&&. (not <$> register False rmsNormValid)

    pendingInput = register False nextPendingInput
     where
      nextPendingInput =
        mux effectiveValidIn (pure False) $
        mux validInRise (pure True)
        pendingInput

    --------------------------------------------------------------------------
    -- BRAM read address: rmsNorm phase uses rdNext; residual phase uses resCounter
    --------------------------------------------------------------------------
    rmsNormRdAddr = (slot2BramBase +) . fromIntegral <$> rdNext

    --------------------------------------------------------------------------
    -- DRAM-backed FFN core: W1 (gate) -> W3 (up) -> W2 (down)
    -- W2 results are written element-by-element to FFN BRAM slot C inside
    -- ffnProjector.  The residual FSM below reads them back via ffnCRdAddr /
    -- ffnBramCRdData after coreValidOut fires.
    --
    -- readyIn is gated (projectorReadyIn) so ffnProjector stays in FPDone
    -- until the residual FSM has finished reading all slot-C elements.
    -- This prevents the projector from starting the next token and overwriting
    -- slot C before we are done reading it.
    --------------------------------------------------------------------------
    resIdle = register True $
      mux coreValidOutRise (pure False) $
      mux writeDone        (pure True)
      resIdle

    -- Guard also against the one-cycle gap on the rising edge of coreValidOut:
    -- resIdle is a register so it still reads True at cycle A when
    -- coreValidOutRise fires.  Without the extra guard the projector can
    -- transition FPDone → FPIdle that same cycle, switching the FFN BRAM
    -- read mux away from slot C before the residual FSM starts reading.
    projectorReadyIn = readyIn .&&. resIdle .&&. (not <$> coreValidOutRise)

    -- ffnCRdAddr: drives the FFN BRAM slot-C read port inside ffnProjector.
    -- Presented one cycle ahead of when the data is needed (1-cycle BRAM latency).
    ffnCRdAddr = mux resActive resLoadCounter (pure 0)

    (ffnAxiMaster, ffnBramCRdData, coreValidOut, readyOut) =
      ffnProjector cycleCounter dramSlaveIn layerIdx effectiveValidIn projectorReadyIn xHat ffnCRdAddr

    axiMasterOut = Master.axiMasterMux rmsFfnBusy rmsFfnAxiMaster ffnAxiMaster

    --------------------------------------------------------------------------
    -- Sequential residual add FSM
    --   Reads slot 2 from activation BRAM and slot C from FFN BRAM in lock-step.
    --   Writes slot3[i] = slot2[i] + w2[i] to activation BRAM.
    --   After ModelDimension + 2 cycles, writeDone fires.
    --------------------------------------------------------------------------
    coreValidOutRise = coreValidOut .&&. (not <$> register False coreValidOut)

    -- resActive: True for ModelDimension cycles while issuing BRAM reads.
    resActive = register False $
      mux coreValidOutRise (pure True) $
      mux resLoadAtMax     (pure False)
      resActive

    resLoadCounter = register 0 $
      mux resActive (satSucc SatWrap <$> resLoadCounter) (pure 0 :: Signal dom (Index ModelDimension))

    resLoadAtMax = resActive .&&. resLoadCounter .==. pure maxBound

    -- resDrain: one extra cycle after resActive ends to capture the last element.
    resDrain = register False resLoadAtMax

    writeDone = register False resDrain

    -- Slot 2 read address during the residual phase.
    resRdAddr = (slot2BramBase +) . fromIntegral <$> resLoadCounter

    -- Time-multiplex the single BRAM read port between rmsNorm and residual phases.
    -- Both phases are strictly sequential (rmsNorm ends before coreValidOut fires).
    bramRdAddr = mux resActive resRdAddr rmsNormRdAddr

    -- Write path: one cycle behind the read (BRAM latency).
    prevResCounter = register (0 :: Index ModelDimension) resLoadCounter

    prevResActive = register False resActive

    -- Write condition: previous cycle was an active load OR we're in the drain cycle.
    inWritePhase = prevResActive .||. resDrain

    slot3WrAddr = (slot3BramBase +) . fromIntegral <$> prevResCounter

    -- ffnBramCRdData delivers w2[prevResCounter] with 1-cycle FFN BRAM latency,
    -- aligned with bramRdData = slot2[prevResCounter] from the activation BRAM.
    bramWrite = mux inWritePhase
      (Just <$> ((,) <$> slot3WrAddr <*> ((+) <$> bramRdData <*> ffnBramCRdData)))
      (pure Nothing)
