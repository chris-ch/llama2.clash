-- File: LLaMa2/Layer/TransformerLayer.hs
module LLaMa2.Layer.TransformerLayer
  ( transformerLayer )
where

import Clash.Prelude
import LLaMa2.Layer.Attention.FSM (processingControllerFSM)
import qualified LLaMa2.Layer.FeedForward.FeedForwardNetwork as FeedForwardNetwork (feedForwardStage)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.LayerData (LayerData (..))
import LLaMa2.Types.ModelConfig
  ( HeadDimension, ModelDimension, NumKeyValueHeads, NumQueryHeads, SequenceLength, NumLayers)
import LLaMa2.Layer.Attention.MultiHeadAttention (multiHeadAttentionStage)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Arbiter as ARB
import LLaMa2.Layer.Attention.QueryHeadProjector (QHeadDebugInfo)
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec)

ffnController ::
  (HiddenClockResetEnable dom) =>
  Signal dom (Unsigned 32) ->
  Slave.AxiSlaveIn dom ->
  Index NumLayers ->
  Signal dom Bool ->
  Signal dom Bool ->
  Signal dom (Vec ModelDimension FixedPoint) ->
  ( Master.AxiMasterOut dom
  , Signal dom (Vec ModelDimension FixedPoint)
  , Signal dom Bool
  , Signal dom Bool
  )
ffnController cycleCounter dramSlaveIn layerIdx inValid outReady inputVec =
  (ffnAxiMaster, result, validOut, inReady)
  where
    (enable, validOut, inReady) = processingControllerFSM inValid outReady ffnSeqValid
    (ffnAxiMaster, result, ffnSeqValid, _readyOut) =
      FeedForwardNetwork.feedForwardStage
        cycleCounter dramSlaveIn layerIdx enable outReady inputVec

transformerLayer ::
  forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  )
   => Signal dom (Unsigned 32)
   -> Slave.AxiSlaveIn dom                              -- ^ weights DRAM
   -> Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)       -- ^ KV cache DRAM (one per KV head)
   -> Index NumLayers
   -> Signal dom (Index SequenceLength)
   -> Signal dom LayerData
   -> Signal dom Bool
   -> ( Master.AxiMasterOut dom                         -- ^ weights AXI master
      , Vec NumKeyValueHeads (Master.AxiMasterOut dom)  -- ^ KV cache AXI masters
      , Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
      , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
      , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
      , Signal dom (Vec ModelDimension FixedPoint)
      , Signal dom (Vec ModelDimension FixedPoint)
      , Signal dom Bool
      , Signal dom Bool
      , Signal dom Bool
      , Signal dom Bool
      , Signal dom Bool
      , QHeadDebugInfo dom
      , Signal dom Bool
      , Signal dom Bool
      , Signal dom Bool
      )
transformerLayer cycleCounter dramSlaveIn kvDramSlaves layerIdx seqPos layerData validIn =
  ( axiMasterOut
  , kvAxiMasters
  , qProj
  , kProj
  , vProj
  , xAfterAttn
  , ffnOutput
  , qkvDone
  , writeDone
  , attentionDone
  , ffnDone
  , qkvReady
  , debugInfo
  , ffnArmed
  , ffnStageStart
  , ffnValidIn
  )
  where
    -- Feed-forward stage busy flag
    layerBusy = register False nextLayerBusy
      where
        nextLayerBusy = mux validInGated (pure True)
                       (mux ffnDone (pure False) layerBusy)

    validInGated = validIn .&&. (not <$> layerBusy)

    -- MHA uses its own weights DRAM slave (from 2-master arbiter)
    (mhaAxiMaster, kvAxiMasters, xAfterAttn, qProj, kProj, vProj, qkvReady, qkvDone, writeDone, attentionDone, debugInfo) =
      multiHeadAttentionStage cycleCounter mhaSlave kvDramSlaves layerIdx seqPos layerData validInGated

    -- 2-master arbiter for weights DRAM: slot 0 = MHA, slot 1 = FFN
    (axiMasterOut, perLayerSlaves) =
      ARB.axiArbiterWithRouting dramSlaveIn
        (mhaAxiMaster :> ffnAxiMaster :> Nil)

    mhaSlave = perLayerSlaves !! (0 :: Index 2)
    ffnSlave = perLayerSlaves !! (1 :: Index 2)

    ffnArmed :: Signal dom Bool
    ffnArmed = register False nextFfnArmed
      where
        setArm   = validInGated
        clearArm = ffnDone
        nextFfnArmed = mux setArm (pure True) (mux clearArm (pure False) ffnArmed)

    ffnStageStart :: Signal dom Bool
    ffnStageStart = attentionDone .&&. ffnArmed

    ffnInput = attentionOutput <$> layerData

    ffnValidIn :: Signal dom Bool
    ffnValidIn = register False nextFFNValidIn
      where
        setFFNValid   = ffnStageStart
        clearFFNValid = ffnValidIn .&&. ffnDone
        nextFFNValidIn =
          mux setFFNValid (pure True)
            (mux clearFFNValid (pure False) ffnValidIn)

    ffnOutReady :: Signal dom Bool
    ffnOutReady = pure True

    (ffnAxiMaster, ffnOutput, ffnValidOut, _ffnInReady) =
      ffnController cycleCounter ffnSlave layerIdx ffnValidIn ffnOutReady ffnInput

    ffnDone = risingEdge ffnValidOut

--------------------------------------------------------------------------------
-- Helper
--------------------------------------------------------------------------------
risingEdge :: HiddenClockResetEnable dom => Signal dom Bool -> Signal dom Bool
risingEdge sig = sig .&&. (not <$> register False sig)
