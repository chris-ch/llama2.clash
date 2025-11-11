module LLaMa2.Layer.TransformerLayer
  ( transformerLayer )
where

import Clash.Prelude
import LLaMa2.Layer.Attention.FSM (processingControllerFSM)
import qualified LLaMa2.Layer.FeedForward.FeedForwardNetwork as FeedForwardNetwork (feedForwardStage)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Types.LayerData
  ( LayerData (..),
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

--------------------------------------------------------------------------------
-- Feed-forward controller
--------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------
-- Transformer layer with stage-local outputs, no internal state mutation
--------------------------------------------------------------------------------
transformerLayer ::
  forall dom.
  (HiddenClockResetEnable dom)
   => PARAM.TransformerLayerComponent
   -> Signal dom (Index SequenceLength)
   -> Signal dom LayerData                -- input layer data
   -> Signal dom QKVProjectionWeightBuffer
   -> Signal dom Bool                     -- validIn (layer-specific)
   -> ( Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))  -- qProj
      , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)) -- kProj
      , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)) -- vProj
      , Signal dom (Vec ModelDimension FixedPoint)  -- attention output
      , Signal dom (Vec ModelDimension FixedPoint)  -- feed-forward output
      , Signal dom Bool  -- qkvDone
      , Signal dom Bool  -- writeDone
      , Signal dom Bool  -- attentionDone
      , Signal dom Bool  -- ffnDone
      , Signal dom Bool  -- qkvReady
      )
transformerLayer layerParams seqPos layerData weightBuffer validIn =
  ( qProj
  , kProj
  , vProj
  , xAfterAttn
  , ffnOutput
  , qkvDone
  , writeDone
  , attentionDone
  , ffnDone
  , qkvReady
  )
  where
    mhaParams = PARAM.multiHeadAttention layerParams
    ffnParams = PARAM.feedforwardNetwork layerParams

    ----------------------------------------------------------------------------
    -- Multi-head attention stage
    ----------------------------------------------------------------------------
    (xAfterAttn, qProj, kProj, vProj, qkvReady, qkvDone, writeDone, attentionDone) =
      multiHeadAttentionStage mhaParams seqPos layerData weightBuffer validIn

    -- ----------------------------------------------------------------------------
    -- Feed-forward stage (Stage 4)
    -- ----------------------------------------------------------------------------

    -- latch that we are inside a valid transaction for this layer
    -- set when validIn asserts, cleared when the FFN finishes for this transaction
    ffnArmed :: Signal dom Bool
    ffnArmed = register False nextFfnArmed
      where
        setArm = validIn                              -- transaction begins
        clearArm = ffnDone                            -- transaction ends when FFN done
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
