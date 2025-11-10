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

--------------------------------------------------------------------------------
-- Simplified feed-forward controller (unchanged)
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
   -> Signal dom CycleStage
   -> Signal dom LayerData                -- input layer data
   -> Signal dom QKVProjectionWeightBuffer
   -> Signal dom Bool                     -- useRAM
   -> Signal dom Bool                     -- validIn (layer-specific)
   -> ( Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))  -- qProj
      , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)) -- kProj
      , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)) -- vProj
      , Signal dom (Vec ModelDimension FixedPoint)  -- attention output
      , Signal dom (Vec ModelDimension FixedPoint)  -- feed-forward output
      , Signal dom Bool  -- writeDone
      , Signal dom Bool  -- attentionDone
      , Signal dom Bool  -- qkvDone
      , Signal dom Bool  -- ffnDone
      )
transformerLayer layerParams seqPos cycleStage layerData weightBuffer useRAM validIn =
  ( qProj
  , kProj
  , vProj
  , xAfterAttn
  , ffnOutput
  , writeDone
  , attentionDone
  , qkvDone
  , ffnDone
  )
  where
    mhaParams = PARAM.multiHeadAttention layerParams
    ffnParams = PARAM.feedforwardNetwork layerParams

    ----------------------------------------------------------------------------
    -- Multi-head attention stage
    ----------------------------------------------------------------------------
    (attentionDone, xAfterAttn, qProj, kProj, vProj, _qkvReady, writeDone, qkvDone) =
      multiHeadAttentionStage mhaParams seqPos layerData weightBuffer useRAM validIn

    ----------------------------------------------------------------------------
    -- Feed-forward stage (Stage 4)
    ----------------------------------------------------------------------------
    inFFNStage = cycleStage .==. pure Stage4_FeedForward
    ffnStageStart = risingEdge inFFNStage

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
    ffnOutReady = pure True

    (ffnOutput, ffnValidOut, ffnInReady) =
      ffnController ffnValidIn ffnOutReady ffnInput ffnParams

    ffnDone = risingEdge ffnValidOut

--------------------------------------------------------------------------------
-- Helper
--------------------------------------------------------------------------------
risingEdge :: HiddenClockResetEnable dom => Signal dom Bool -> Signal dom Bool
risingEdge sig = sig .&&. (not <$> register False sig)
