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
import LLaMa2.Types.Parameters (FeedForwardNetworkComponentQ, TransformerLayerComponent (..))
import LLaMa2.Layer.Attention.MultiHeadAttention (multiHeadAttentionStage)
import LLaMa2.Layer.Attention.WeightBuffer (QKVWeightBuffer(..))

ffnController ::
  (HiddenClockResetEnable dom) =>
  Signal dom Bool ->
  Signal dom Bool ->
  Signal dom (Vec ModelDimension FixedPoint) ->
  FeedForwardNetworkComponentQ ->
  ( Signal dom (Vec ModelDimension FixedPoint),
    Signal dom Bool,
    Signal dom Bool
  )
ffnController inValid outReady inputVec ffnQ = (result, validOut, inReady)
  where
    (enable, validOut, inReady) = processingControllerFSM inValid outReady ffnSeqValid
    (result, ffnSeqValid, _ready) =
      FeedForwardNetwork.feedForwardStage enable outReady ffnQ inputVec

transformerLayer ::
  forall dom.
  (HiddenClockResetEnable dom)
   => TransformerLayerComponent
   -> Index NumLayers
   -> Signal dom ProcessingState
   -> Signal dom LayerData
   -> Signal dom QKVWeightBuffer             -- full RAM buffer
   -> Signal dom Bool                        --
   -> ( Signal dom LayerData,
    Signal dom Bool, -- writeDone
    Signal dom Bool, -- attentionDone
    Signal dom Bool, -- qkvDone
    Signal dom LayerData, -- layerDataAfterAttention
    Signal dom Bool, -- qkvInReady
    Signal dom Bool -- ffnDone
  )
transformerLayer layer layerIndex processingState layerData weightBuffer enable =
  ( nextLayerData,
    writeDone,
    attentionDone,
    qkvDone,
    layerDataAfterAttention,
    qkvInReady,
    ffnValidOut
  )
  where
    mha = multiHeadAttention layer
    ffn = feedforwardNetwork layer

    (attentionDone, xAfterAttn, qProj, kProj, vProj, qkvInReady, writeDone, qkvDone) =
      multiHeadAttentionStage mha processingState layerIndex layerData weightBuffer enable

    layerDataAfterAttention =
      (layerDataAttnDone layerIndex <$> processingState)
        <*> layerData
        <*> xAfterAttn
        <*> attentionDone

    baseNextLayerData =
      updateLayerDataForStage layerIndex <$> processingState <*> layerData <*> qProj <*> kProj <*> vProj

    -- Stage 4 FFN
    isStage4ThisLayer =
      ( \ps ->
          processingStage ps
            == Stage4_FeedForward
            && processingLayer ps
            == layerIndex
      )
        <$> processingState
    ffnInput = attentionOutput <$> layerDataAfterAttention
    ffnOutReady =
      ( \ps -> case () of
          _
            | processingStage ps
                == Stage1_ProjectQKV
                && processingLayer ps
                == layerIndex
                + 1 ->
                True
            | processingStage ps
                == Stage5_Classifier
                && processingLayer ps
                == maxBound ->
                True
            | otherwise -> False
      )
        <$> processingState
    (ffnOutput, ffnValidOut, _ffnInReady) =
      ffnController
        isStage4ThisLayer
        ffnOutReady
        ffnInput
        ffn
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
