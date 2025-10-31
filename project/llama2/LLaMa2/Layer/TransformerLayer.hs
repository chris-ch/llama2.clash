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
   -> Signal dom QKVProjectionWeightBuffer
   -> Signal dom Bool
   -> Signal dom Bool  -- enableQKV (layer-specific)
   -> Signal dom Bool  -- enableWriteKV (layer-specific)
   -> Signal dom Bool  -- enableAttend (layer-specific)
   -> Signal dom Bool  -- enableFFN (layer-specific)
   -> ( Signal dom LayerData,
        Signal dom Bool,
        Signal dom Bool,
        Signal dom Bool,
        Signal dom Bool
      )
transformerLayer layer layerIndex processingState layerData weightBuffer useRAM
                 enableQKV enableWriteKV enableAttend enableFFN =
  ( nextLayerData,
    writeDone,
    attentionDone,
    qkvDone,
    ffnValidOut
  )
  where
    mha = PARAM.multiHeadAttention layer
    ffn = PARAM.feedforwardNetwork layer

    seqPos = sequencePosition <$> processingState

    -- Enables are already layer-specific from SimplifiedLayerStack
    (attentionDone, xAfterAttn, qProj, kProj, vProj, qkvInReady, writeDone, qkvDone) =
      multiHeadAttentionStage mha seqPos layerData weightBuffer useRAM
                              enableQKV enableWriteKV enableAttend

    layerDataAfterAttention :: Signal dom LayerData
    layerDataAfterAttention =
      (layerDataAttnDone layerIndex <$> processingState)
        <*> layerData
        <*> xAfterAttn
        <*> attentionDone

    baseNextLayerData :: Signal dom LayerData
    baseNextLayerData =
      updateLayerDataForStage layerIndex <$> processingState <*> layerData <*> qProj <*> kProj <*> vProj

    -- FFN stage - enableFFN is already layer-specific
    ffnInput = attentionOutput <$> layerDataAfterAttention

    -- FFN output ready when:
    -- - Next layer is starting QKV (need to check which layer is active)
    -- - Or we're at last layer and classifier is starting
    ffnOutReady :: Signal dom Bool
    ffnOutReady =
      -- Check if we're ready for next layer or classifier
      -- Note: These checks use currentLayer to coordinate between layers
      (((processingStage <$> processingState) .==. pure Stage1_ProjectQKV)
        .&&.
      ((processingLayer <$> processingState) .==. pure (layerIndex + 1)))
      .||.
      (((processingStage <$> processingState) .==. pure Stage5_Classifier)
        .&&.
      ((processingLayer <$> processingState) .==. pure maxBound))

    (ffnOutput, ffnValidOut, ffnInReady) =
      ffnController
        enableFFN       -- Already layer-specific
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

-- Helper functions remain unchanged
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
   in if processingLayer ps == layerIndex
        && processingStage ps == Stage4_FeedForward
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
  if processingLayer stage == layerIndex
    && processingStage stage == Stage3_Attend
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
