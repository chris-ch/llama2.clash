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

-- | Transformer layer with simplified stage interface
-- Now receives: enableAttention, enableFFN, enableClassifier
-- Returns: attentionDone (single signal for entire attention mechanism), ffnDone
transformerLayer ::
  forall dom.
  (HiddenClockResetEnable dom)
   => PARAM.TransformerLayerComponent
   -> Index NumLayers
   -> Signal dom ProcessingState
   -> Signal dom LayerData
   -> Signal dom QKVProjectionWeightBuffer
   -> Signal dom Bool              -- useRAM
   -> Signal dom Bool              -- enableAttention (layer-specific)
   -> Signal dom Bool              -- enableFFN (layer-specific)
   -> Signal dom Bool              -- enableClassifier (layer-specific)
   -> ( Signal dom LayerData,      -- nextLayerData
        Signal dom Bool,           -- attentionDone (replaces qkvDone, writeDone, attnDone)
        Signal dom Bool            -- ffnDone (rising edge only)
      )
transformerLayer layer layerIndex processingState layerData weightBuffer useRAM
                 enableAttention enableFFN enableClassifier =
  ( nextLayerData,
    attentionDone,
    ffnDoneEdge  -- Use edge detector instead of raw validOut
  )
  where
    mha = PARAM.multiHeadAttention layer
    ffn = PARAM.feedforwardNetwork layer

    seqPos = sequencePosition <$> processingState

    -- ==========================================================================
    -- Attention Stage (self-contained, manages QKV+Write+Attend internally)
    -- ==========================================================================
    -- MultiHeadAttention now returns Q/K/V values that are already latched internally
    (attentionDone, xAfterAttn, qProj, kProj, vProj) =
      multiHeadAttentionStage mha seqPos layerData weightBuffer useRAM enableAttention

    -- Update layer data when attention completes
    layerDataAfterAttention :: Signal dom LayerData
    layerDataAfterAttention =
      (layerDataAttnDone layerIndex <$> processingState)
        <*> layerData
        <*> xAfterAttn
        <*> attentionDone

    -- Update layer data with QKV projections (already latched by MultiHeadAttention)
    baseNextLayerData :: Signal dom LayerData
    baseNextLayerData =
      (\ld q k v -> ld { queryVectors = q
                       , keyVectors = k
                       , valueVectors = v
                       }) <$> layerData <*> qProj <*> kProj <*> vProj

    -- ==========================================================================
    -- Feed-Forward Stage
    -- ==========================================================================
    ffnInput = attentionOutput <$> layerDataAfterAttention

    -- FFN output ready when:
    -- - Next layer is starting Attention (need to check which layer is active)
    -- - Or we're at last layer and classifier is starting
    ffnOutReady :: Signal dom Bool
    ffnOutReady =
      let currentLayer = processingLayer <$> processingState
      in (enableAttention .&&. (currentLayer .==. pure (layerIndex + 1)))
        .||.
        (enableClassifier .&&. (currentLayer .==. pure maxBound))

    (ffnOutput, ffnValidOut, ffnInReady) =
      ffnController
        enableFFN
        ffnOutReady
        ffnInput
        ffn

    -- CRITICAL FIX: Detect rising edge of ffnValidOut to avoid multiple completions
    ffnDoneEdge = risingEdge ffnValidOut
      where
        risingEdge sig = sig .&&. (not <$> register False sig)

    -- Update layer data with FFN output
    nextLayerData :: Signal dom LayerData
    nextLayerData =
      (layerDataWithFFN layerIndex <$> processingState)
        <*> baseNextLayerData
        <*> xAfterAttn
        <*> attentionDone
        <*> ffnOutput
        <*> ffnValidOut

-- Helper functions
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
   in if processingLayer ps == layerIndex && ffnValid
        -- Don't check stage! Just store when ffnValid pulses and it's our layer.
        then withAttn {feedForwardOutput = ffnOut}
        else withAttn

layerDataAttnDone ::
  Index NumLayers ->
  ProcessingState ->
  LayerData ->
  Vec ModelDimension FixedPoint ->
  Bool ->
  LayerData
layerDataAttnDone layerIndex state cur attOut attnDone =
  -- Don't check stage! The stage may have already advanced.
  -- Just store when attnDone pulses and it's our layer.
  if processingLayer state == layerIndex && attnDone
    then cur {attentionOutput = attOut}
    else cur
