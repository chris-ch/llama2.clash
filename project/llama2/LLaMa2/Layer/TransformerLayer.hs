-- File: LLaMa2/Layer/TransformerLayer.hs (add AXI threading)
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
import qualified Simulation.Parameters as PARAM (FeedForwardNetworkComponentQ, TransformerLayerComponent (..), DecoderParameters)
import LLaMa2.Layer.Attention.MultiHeadAttention (multiHeadAttentionStage)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import Simulation.Parameters (DecoderParameters(..))
import LLaMa2.Layer.Attention.QueryHeadProjector (QHeadDebugInfo)

ffnController ::
  (HiddenClockResetEnable dom) =>
  Signal dom Bool ->
  Signal dom Bool ->
  Signal dom (Vec ModelDimension FixedPoint) ->
  PARAM.FeedForwardNetworkComponentQ ->
  ( Signal dom (Vec ModelDimension FixedPoint),
    Signal dom Bool,
    Signal dom Bool
  )
ffnController inValid outReady inputVec ffnQ = (result, validOut, inReady)
  where
    (enable, validOut, inReady) = processingControllerFSM inValid outReady ffnSeqValid
    (result, ffnSeqValid, _) =
      FeedForwardNetwork.feedForwardStage enable outReady ffnQ inputVec

transformerLayer ::
  forall dom.
  (HiddenClockResetEnable dom)
   => Slave.AxiSlaveIn dom                   -- DRAM interface
   -> Index NumLayers                        -- layer index
   -> PARAM.DecoderParameters
   -> Signal dom (Index SequenceLength)
   -> Signal dom LayerData
   -> Signal dom Bool
   -> ( Master.AxiMasterOut dom -- AXI master out
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
transformerLayer dramSlaveIn layerIdx params seqPos layerData validIn =
  ( axiMasterOut
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
    layerParams = modelLayers params !! layerIdx

    ffnParams = PARAM.feedforwardNetwork layerParams

    -- Feed-forward stage
    layerBusy = register False nextLayerBusy
      where
        nextLayerBusy = mux validInGated (pure True)
                       (mux ffnDone (pure False) layerBusy)

    validInGated = validIn .&&. (not <$> layerBusy)

    -- Use validInGated everywhere instead of validIn
    (axiMasterOut, xAfterAttn, qProj, kProj, vProj, qkvReady, qkvDone, writeDone, attentionDone, debugInfo) =
      multiHeadAttentionStage dramSlaveIn layerIdx params seqPos layerData validInGated

    ffnArmed :: Signal dom Bool
    ffnArmed = register False nextFfnArmed
      where
        setArm = validInGated  -- Use gated version
        clearArm = ffnDone
        nextFfnArmed = mux setArm (pure True) (mux clearArm (pure False) ffnArmed)

    -- ffnStageStart: only start FFN when attentionDone *and* we are armed for this layer
    -- note: attentionDone is already a pulse; no extra risingEdge is required
    ffnStageStart :: Signal dom Bool
    ffnStageStart = attentionDone .&&. ffnArmed

    ffnInput = attentionOutput <$> layerData

    -- Convert start pulse to sustained valid
    ffnValidIn :: Signal dom Bool
    ffnValidIn = register False nextFFNValidIn
      where
        setFFNValid   = ffnStageStart
        clearFFNValid = ffnValidIn .&&. ffnInReady
        nextFFNValidIn =
          mux setFFNValid (pure True)
            (mux clearFFNValid (pure False) ffnValidIn)

    ffnOutReady :: Signal dom Bool
    ffnOutReady = pure True  -- always ready (?)

    (ffnOutput, ffnValidOut, ffnInReady) =
      ffnController ffnValidIn ffnOutReady ffnInput ffnParams

    ffnDone = risingEdge ffnValidOut

--------------------------------------------------------------------------------
-- Helper
--------------------------------------------------------------------------------
risingEdge :: HiddenClockResetEnable dom => Signal dom Bool -> Signal dom Bool
risingEdge sig = sig .&&. (not <$> register False sig)
