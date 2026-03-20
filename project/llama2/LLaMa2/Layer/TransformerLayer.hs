-- File: LLaMa2/Layer/TransformerLayer.hs
module LLaMa2.Layer.TransformerLayer
  ( transformerLayer )
where

import Clash.Prelude
import qualified GHC.TypeNats as TN
import qualified LLaMa2.Layer.FeedForward.FeedForwardNetwork as FeedForwardNetwork (feedForwardStage)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.LayerData (ActivationBramAddr)
import LLaMa2.Types.ModelConfig
  ( HeadDimension, ModelDimension, NumKeyValueHeads, SequenceLength, NumLayers)
import LLaMa2.Layer.Attention.MultiHeadAttention (multiHeadAttentionStage)
import LLaMa2.Memory.ActivationBRAM
  ( ActivationBramReadPort (..), ActivationBramWritePort (..), activationBram )
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Arbiter as ARB
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec)

slot0BramBase :: ActivationBramAddr
slot0BramBase = 0

slot3BramBase :: ActivationBramAddr
slot3BramBase = natToNum @(3 TN.* ModelDimension)

-- | One transformer layer (MHA → FFN → slot3-to-slot0 copy).
--
-- The activation BRAM is instantiated internally.
-- Slot layout:
--   0 = inputVector   (written externally via @initWrPort@, or by the copy phase)
--   2 = attnOutput    (written by MHA residual adder)
--   3 = ffnOutput     (written by FFN residual adder)
--
-- After FFN completes, an internal copy phase streams slot 3 → slot 0 to
-- prepare the next layer run.  When the layer is idle, the activation BRAM
-- read port is handed to the caller via @extBramRdAddr@ / @bramRdDataOut@
-- so the logits projector can read slot 3 directly.
--
-- @initWrPort@ (lowest-priority write) is used by the caller to inject the
-- token embedding into slot 0 for layer 0.  It is safe to drive @initWrPort@
-- whenever the layer is idle (@readyOut = True@).
transformerLayer ::
  forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  )
   => Signal dom (Unsigned 32)
   -> Slave.AxiSlaveIn dom                                    -- ^ weights DRAM
   -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)             -- ^ KV cache DRAM
   -> Signal dom (Index NumLayers)
   -> Signal dom (Index SequenceLength)
   -> Signal dom (Maybe (ActivationBramAddr, FixedPoint))     -- ^ slot 0 init write
   -> Signal dom Bool                                         -- ^ validIn
   -> Signal dom ActivationBramAddr                           -- ^ extBramRdAddr (when layer idle)
   -> ( Master.AxiMasterOut dom
      , Vec NumKeyValueHeads (Master.AxiMasterOut dom)
      , Signal dom Bool                                       -- ^ layerDone (copy complete)
      , Signal dom Bool                                       -- ^ readyOut (layer idle)
      , Signal dom FixedPoint                                 -- ^ bramRdDataOut (activation BRAM)
      , Signal dom FixedPoint                                 -- ^ ffnOut0 debug: slot3[0] captured during copy
      )
transformerLayer cycleCounter dramSlaveIn kvDramSlaves layerIdx seqPos initWrPort validIn extBramRdAddr =
  (axiMasterOut, kvAxiMasters, layerDone, readyOut, bramRdData, ffnOut0Debug)
  where
    --------------------------------------------------------------------------
    -- Phase tracking (strictly sequential: mha → ffn → copy → idle)
    --------------------------------------------------------------------------
    mhaPhaseActive = register False $
      mux mhaWriteDone (pure False) $
      mux validIn (pure True)
      mhaPhaseActive

    mhaWriteDonePulse = mhaWriteDone .&&. (not <$> register False mhaWriteDone)

    ffnPhaseActive = register False $
      mux ffnWriteDone (pure False) $
      mux mhaWriteDonePulse (pure True)
      ffnPhaseActive

    ffnWriteDonePulse = ffnWriteDone .&&. (not <$> register False ffnWriteDone)

    copyActive = register False $
      mux copyDone (pure False) $
      mux ffnWriteDonePulse (pure True)
      copyActive

    copyCounter = register (0 :: Index ModelDimension) $
      mux copyActive (satSucc SatWrap <$> copyCounter) (pure 0)

    copyAtMax = copyActive .&&. copyCounter .==. pure maxBound

    copyDrain = register False copyAtMax

    layerDone = register False copyDrain

    copyDone = layerDone  -- one cycle after drain

    --------------------------------------------------------------------------
    -- Activation BRAM — read address mux
    -- Priority: copy > ffn > mha (strictly sequential, no contention)
    --------------------------------------------------------------------------
    slot3RdAddr   = (slot3BramBase +) . fromIntegral <$> copyCounter
    bramRdAddr = mux copyActive slot3RdAddr $
                 mux ffnPhaseActive ffnBramRdAddr $
                 mux mhaPhaseActive mhaBramRdAddr
                 extBramRdAddr   -- when idle: external caller drives the read port

    --------------------------------------------------------------------------
    -- Activation BRAM — write mux
    -- During copy: write slot 0; during stage phases: stage writes;
    -- when idle: external initWrPort (embedding for layer 0).
    --------------------------------------------------------------------------
    prevCopyCounter = register (0 :: Index ModelDimension) copyCounter
    prevCopyActive  = register False copyActive
    inCopyWritePhase = prevCopyActive .||. copyDrain

    copyWrAddr = (slot0BramBase +) . fromIntegral <$> prevCopyCounter
    copyWrite  = mux inCopyWritePhase
      (Just <$> ((,) <$> copyWrAddr <*> bramRdData))
      (pure Nothing)

    activeWrite = mux (copyActive .||. copyDrain) copyWrite $
                  mux ffnPhaseActive ffnBramWrite $
                  mux mhaPhaseActive mhaBramWrite
                  initWrPort

    bramWrAddr = maybe 0 fst <$> activeWrite
    bramWrData = fmap snd   <$> activeWrite

    bramRdData = activationBram
      (ActivationBramReadPort  bramRdAddr)
      (ActivationBramWritePort bramWrAddr bramWrData)

    --------------------------------------------------------------------------
    -- 2-master AXI arbiter: slot 0 = MHA weights, slot 1 = FFN weights
    --------------------------------------------------------------------------
    (axiMasterOut, perLayerSlaves) =
      ARB.axiArbiterWithRouting dramSlaveIn
        (mhaAxiMaster :> ffnAxiMaster :> Nil)

    mhaSlave = perLayerSlaves !! (0 :: Index 2)
    ffnSlave = perLayerSlaves !! (1 :: Index 2)

    --------------------------------------------------------------------------
    -- Multi-head attention (reads slot 0, writes slot 2)
    --------------------------------------------------------------------------
    (mhaAxiMaster, kvAxiMasters, mhaWriteDone, _mhaReadyOut, mhaBramRdAddr, mhaBramWrite) =
      multiHeadAttentionStage
        cycleCounter mhaSlave kvDramSlaves layerIdx seqPos validIn bramRdData

    --------------------------------------------------------------------------
    -- Feed-forward stage (reads slot 2, writes slot 3)
    --------------------------------------------------------------------------
    (ffnAxiMaster, ffnWriteDone, _ffnReadyOut, ffnBramRdAddr, ffnBramWrite) =
      FeedForwardNetwork.feedForwardStage
        cycleCounter ffnSlave layerIdx mhaWriteDone (pure True) bramRdData

    readyOut = (not <$> mhaPhaseActive) .&&. (not <$> ffnPhaseActive) .&&. (not <$> copyActive)

    -- Debug: capture slot3[0] during copy phase (copyCounter==1 means slot3[0] data has arrived).
    ffnOut0Debug = regEn 0 (copyActive .&&. copyCounter .==. pure 1) bramRdData
