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
    NumQueryHeads, SequenceLength,
  )
import qualified Simulation.Parameters as PARAM (FeedForwardNetworkComponentQ, TransformerLayerComponent (..))
import LLaMa2.Layer.Attention.MultiHeadAttention (multiHeadAttentionStage)
import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer (QKVProjectionWeightBuffer(..))

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
    ffnDone  -- Now a proper pulse!
  )
  where
    mha = PARAM.multiHeadAttention layerParams
    ffn = PARAM.feedforwardNetwork layerParams

    (attentionDone, xAfterAttn, qProj, kProj, vProj, qkvReady, writeDone, qkvDone) =
      multiHeadAttentionStage mha seqPos layerData weightBuffer useRAM validIn 

    layerDataAfterAttention :: Signal dom LayerData
    layerDataAfterAttention = layerDataAttnDone <$> cycleStage
        <*> layerData
        <*> xAfterAttn

    baseNextLayerData :: Signal dom LayerData
    baseNextLayerData = layerDataQKV <$> cycleStage <*> layerData <*> qProj <*> kProj <*> vProj
    
    -- Detect rising edge of Stage4_FeedForward for this layer
    inFFNStage = cycleStage .==. pure Stage4_FeedForward
    ffnStageStart = risingEdge inFFNStage
    
    ffnInput = attentionOutput <$> layerDataAfterAttention

    -- Convert stage start pulse to sustained valid signal for FFN
    ffnValidIn :: Signal dom Bool
    ffnValidIn = register False nextFFNValidIn
      where
        setFFNValid = ffnStageStart
        clearFFNValid = ffnValidIn .&&. ffnInReady
        nextFFNValidIn = 
          mux setFFNValid (pure True) 
            (mux clearFFNValid (pure False) ffnValidIn)

    -- FFN output is always ready to be consumed by the sequencer
    ffnOutReady :: Signal dom Bool
    ffnOutReady = pure True

    (ffnOutput, ffnValidOut, ffnInReady) =
      ffnController
        ffnValidIn  -- Triggered by entering Stage4_FeedForward
        ffnOutReady -- Always ready
        ffnInput
        ffn

    -- Convert ffnValidOut to a pulse (rising edge detection)
    ffnDone :: Signal dom Bool
    ffnDone = risingEdge ffnValidOut

    nextLayerData :: Signal dom LayerData
    nextLayerData = layerDataWithFFN <$> cycleStage
        <*> baseNextLayerData
        <*> xAfterAttn
        <*> ffnOutput

-- Helper function for rising edge detection
risingEdge :: HiddenClockResetEnable dom => Signal dom Bool -> Signal dom Bool
risingEdge sig = sig .&&. (not <$> register False sig)

-- Helper functions
layerDataWithFFN :: CycleStage ->
  LayerData ->
  Vec ModelDimension FixedPoint ->
  Vec ModelDimension FixedPoint ->
  LayerData
layerDataWithFFN stage baseData attnOut ffnOut =
  let withAttn = layerDataAttnDone stage baseData attnOut
   in if stage == Stage4_FeedForward
        then withAttn {feedForwardOutput = ffnOut}
        else withAttn

layerDataAttnDone :: CycleStage ->
  LayerData ->
  Vec ModelDimension FixedPoint ->
  LayerData
layerDataAttnDone stage baseData attOut =
  if stage == Stage3_Attend
    then baseData {attentionOutput = attOut}
    else baseData

layerDataQKV :: CycleStage
  -> LayerData 
  -> Vec NumQueryHeads (Vec HeadDimension FixedPoint)
  -> Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  -> Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  -> LayerData
layerDataQKV stage baseData qs ks vs = case stage of
      Stage1_ProjectQKV ->
        baseData {queryVectors = qs, keyVectors = ks, valueVectors = vs}
      _ -> baseData
