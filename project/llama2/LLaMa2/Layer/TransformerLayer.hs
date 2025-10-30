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
    ProcessingState (..),
  )
import LLaMa2.Types.ModelConfig
  ( HeadDimension,
    ModelDimension,
    NumKeyValueHeads,
    NumLayers,
    NumQueryHeads,
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
   -> Index NumLayers
   -> Signal dom ProcessingState
   -> Signal dom LayerData
   -> Signal dom QKVProjectionWeightBuffer             -- full RAM buffer
   -> Signal dom Bool                        --
   -> ( Signal dom LayerData,
    Signal dom Bool, -- writeDone
    Signal dom Bool, -- attentionDone
    Signal dom Bool, -- qkvDone
    Signal dom LayerData, -- layerDataAfterAttention
    Signal dom Bool, -- qkvInReady
    Signal dom Bool -- ffnDone
  )
transformerLayer layer layerIndex processingState layerData weightBuffer useRAM =
  ( nextLayerData,
    writeDone,
    attentionDone,
    qkvDone,
    layerDataAfterAttention,
    qkvInReady,
    ffnValidOut
  )
  where
    mha = PARAM.multiHeadAttention layer
    ffn = PARAM.feedforwardNetwork layer

    xAfterAttn :: Signal dom (Vec ModelDimension FixedPoint)
    qProj :: Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
    kProj :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
    vProj :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
    (attentionDone, xAfterAttn, qProj, kProj, vProj, qkvInReady, writeDone, qkvDone) =
      multiHeadAttentionStage mha processingState layerIndex layerData weightBuffer useRAM

    layerDataAfterAttention :: Signal dom LayerData
    layerDataAfterAttention =
      (layerDataAttnDone layerIndex <$> processingState)
        <*> layerData
        <*> xAfterAttn
        <*> attentionDone

    baseNextLayerData :: Signal dom LayerData
    baseNextLayerData =
      updateLayerDataForStage layerIndex <$> processingState <*> layerData <*> qProj <*> kProj <*> vProj

    -- Stage 4 FFN
    isStage4ThisLayer :: Signal dom Bool
    isStage4ThisLayer =
      ((processingStage <$> processingState) .==. pure Stage4_FeedForward)
      .&&.
      ((processingLayer <$> processingState) .==. pure layerIndex)

    ffnInput = attentionOutput <$> layerDataAfterAttention

    ffnOutReady :: Signal dom Bool
    ffnOutReady =
      (((processingStage <$> processingState) .==. pure Stage1_ProjectQKV)
        .&&.
      ((processingLayer <$> processingState) .==. pure (layerIndex + 1)))
      .||.
      (((processingStage <$> processingState) .==. pure Stage5_Classifier)
        .&&.
      ((processingLayer <$> processingState) .==. pure maxBound))

    ffnOutput :: Signal dom (Vec ModelDimension FixedPoint)
    (ffnOutput, ffnValidOut, ffnInReady) =
      ffnController
        isStage4ThisLayer
        ffnOutReady
        ffnInput
        ffn

    nextLayerData :: Signal dom LayerData
    nextLayerData =
      (layerDataWithFFN layerIndex <$> processingState)
        <*> baseNextLayerData
        <*> xAfterAttn
        <*> attentionDone
        <*> ffnOutput
        <*> ffnValidOut

layerDataWithFFN ::
  Index NumLayers ->
  ProcessingState ->
  LayerData ->
  Vec ModelDimension FixedPoint ->
  Bool ->
  Vec ModelDimension FixedPoint ->
  Bool ->
  LayerData
layerDataWithFFN layerIndex ps baseData attnOut attnDone ffnOut ffnValid =
  let withAttn = layerDataAttnDone layerIndex ps baseData attnOut attnDone
   in if processingLayer ps
        == layerIndex
        && processingStage ps
        == Stage4_FeedForward
        && ffnValid
        then withAttn {feedForwardOutput = ffnOut}
        else withAttn

layerDataAttnDone ::
  Index NumLayers ->
  ProcessingState ->
  LayerData ->
  Vec ModelDimension FixedPoint ->
  Bool ->
  LayerData
layerDataAttnDone layerIndex stage cur attOut attnDone =
  if processingLayer stage
    == layerIndex
    && processingStage stage
    == Stage3_Attend
    && attnDone
    then cur {attentionOutput = attOut}
    else cur

updateLayerDataForStage :: Index NumLayers
  -> ProcessingState
  -> LayerData 
  -> Vec NumQueryHeads (Vec HeadDimension FixedPoint)
  -> Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  -> Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
  -> LayerData
updateLayerDataForStage layerIndex ps idata qs ks vs
  | processingLayer ps /= layerIndex = idata
  | otherwise = case processingStage ps of
      Stage1_ProjectQKV ->
        idata {queryVectors = qs, keyVectors = ks, valueVectors = vs}
      _ -> idata
