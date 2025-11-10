module LLaMa2.Layer.TransformerLayer
  ( transformerLayer )
where

import Clash.Prelude
import LLaMa2.Layer.Attention.FSM (processingControllerFSM)
import qualified LLaMa2.Layer.FeedForward.FeedForwardNetwork as FeedForwardNetwork (feedForwardStage)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.LayerData
  ( CycleStage (..),
    LayerData (..),
  )
import LLaMa2.Types.ModelConfig
  ( HeadDimension,
    ModelDimension,
    NumKeyValueHeads,
    NumQueryHeads,
    SequenceLength,
  )
import qualified Simulation.Parameters as PARAM (FeedForwardNetworkComponentQ, TransformerLayerComponent (..))
import LLaMa2.Layer.Attention.MultiHeadAttention (multiHeadAttentionStage)
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer (QKVProjectionWeightBuffer(..))

-- Define a type for LayerData updates
data LayerUpdate
  = UpdateQKV (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
              (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
              (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  | UpdateAttention (Vec ModelDimension FixedPoint)
  | UpdateFeedForward (Vec ModelDimension FixedPoint)
  | NoUpdate

-- Unified function to update LayerData based on CycleStage and update data
updateLayerData :: CycleStage -> LayerData -> LayerUpdate -> LayerData
updateLayerData stage baseData update = case (stage, update) of
  (Stage1_ProjectQKV, UpdateQKV qs ks vs) ->
    baseData { queryVectors = qs, keyVectors = ks, valueVectors = vs }
  (Stage3_Attend, UpdateAttention attnOut) ->
    baseData { attentionOutput = attnOut }
  (Stage4_FeedForward, UpdateFeedForward ffnOut) ->
    baseData { feedForwardOutput = ffnOut }
  _ -> baseData

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
    (result, ffnSeqValid, ready) =
      FeedForwardNetwork.feedForwardStage enable outReady ffnQ inputVec

transformerLayer ::
  forall dom.
  (HiddenClockResetEnable dom)
   => PARAM.TransformerLayerComponent
   -> Signal dom (Index SequenceLength)
   -> Signal dom CycleStage
   -> Signal dom LayerData
   -> Signal dom QKVProjectionWeightBuffer
   -> Signal dom Bool
   -> Signal dom Bool  -- enableQKV (layer-specific)
   -> ( Signal dom LayerData,
        Signal dom Bool,
        Signal dom Bool,
        Signal dom Bool,
        Signal dom Bool
      )
transformerLayer layerParams seqPos cycleStage layerData weightBuffer useRAM validIn =
  ( nextLayerData,
    writeDone,
    attentionDone,
    qkvDone,
    ffnDone
  )
  where
    mhaParams = PARAM.multiHeadAttention layerParams
    ffnParams = PARAM.feedforwardNetwork layerParams

    -- Multi-head attention stage
    (attentionDone, xAfterAttn, qProj, kProj, vProj, qkvReady, writeDone, qkvDone) =
      multiHeadAttentionStage mhaParams seqPos layerData weightBuffer useRAM validIn

    -- Detect rising edge of Stage4_FeedForward
    inFFNStage = cycleStage .==. pure Stage4_FeedForward
    ffnStageStart = risingEdge inFFNStage

    -- FFN input from attention output
    ffnInput = attentionOutput <$> layerData

    -- Convert stage start pulse to sustained valid signal for FFN
    ffnValidIn :: Signal dom Bool
    ffnValidIn = register False nextFFNValidIn
      where
        setFFNValid = ffnStageStart
        clearFFNValid = ffnValidIn .&&. ffnInReady
        nextFFNValidIn =
          mux setFFNValid (pure True)
            (mux clearFFNValid (pure False) ffnValidIn)

    -- FFN output is always ready to be consumed
    ffnOutReady :: Signal dom Bool
    ffnOutReady = pure True

    (ffnOutput, ffnValidOut, ffnInReady) =
      ffnController
        ffnValidIn
        ffnOutReady
        ffnInput
        ffnParams

    -- Convert ffnValidOut to a pulse
    ffnDone :: Signal dom Bool
    ffnDone = risingEdge ffnValidOut

    -- Determine the appropriate LayerUpdate based on CycleStage
    layerUpdate :: Signal dom LayerUpdate
    layerUpdate = fmap mkLayerUpdate (bundle (cycleStage, qProj, kProj, vProj, xAfterAttn, ffnOutput))
      where
        mkLayerUpdate (stage, q, k, v, attn, ffn) =
          case stage of
            Stage1_ProjectQKV -> UpdateQKV q k v
            Stage3_Attend -> UpdateAttention attn
            Stage4_FeedForward -> UpdateFeedForward ffn
            _ -> NoUpdate

    -- Update LayerData using the unified update function
    nextLayerData :: Signal dom LayerData
    nextLayerData = updateLayerData <$> cycleStage <*> layerData <*> layerUpdate

-- Helper function for rising edge detection
risingEdge :: HiddenClockResetEnable dom => Signal dom Bool -> Signal dom Bool
risingEdge sig = sig .&&. (not <$> register False sig)
