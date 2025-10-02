module Model.Core.Transformer.Debug (
    multiCycleTransformerDebug
) where

import Clash.Prelude
import Helpers (liftA4)
import Model.Config (NumLayers, ModelDimension)

import Model.Core.Transformer.Internal
import Model.Core.Types
  ( LayerData(..)
  , ProcessingState(..)
  , CycleStage(..)
  , Seed, Temperature, Token
  )
import qualified Model.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs(..)
  )
import qualified Model.Layers.TransformerLayer as TransformerLayer
  ( TransformerDecoderComponent(..)
  , TransformerLayerComponent(..)
  )
import qualified Model.Layers.TransformerLayer.Debug as TLDbg
  ( multiCycleTransformerLayerDbg
  )
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Embedding.PRNG as PRNG
import qualified Model.Core.Embedding as Embedding
import qualified Model.Layers.Components.Quantized as Quantized
  ( EmbeddingComponentQ(..) )
import Model.Numeric.ParamPack (QArray2D)
import Model.Numeric.Types (FixedPoint)
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..))

-- | Return (token, readyPulse, kErrVec, vErrVec)
multiCycleTransformerDebug :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , Vec NumLayers (Signal dom Bool)
     , Vec NumLayers (Signal dom Bool)
     )
multiCycleTransformerDebug decoder cacheOwners inputToken inputTokenValid temperature seed =
  (selectedToken, readyPulse, kErrVec, vErrVec)
 where
  vocabulary = Quantized.vocabularyQ $ modelEmbedding decoder

  transformerLayers :: Vec NumLayers TransformerLayer.TransformerLayerComponent
  transformerLayers  = modelLayers decoder

  pipelineController :: PipelineController.PipelineOutputs dom
  pipelineController = PipelineController.runPipelineController attnDoneThisLayer writeDoneThisLayer inputTokenValid

  layerIndex :: Signal dom (Index NumLayers)
  layerIndex = PipelineController.layerIndex pipelineController

  readyPulse :: Signal dom Bool
  readyPulse = PipelineController.readyPulse pipelineController

  processingState :: Signal dom ProcessingState
  processingState = PipelineController.processingState pipelineController

  tokenSample :: Signal dom Token
  tokenSample = PRNG.tokenSampler readyPulse temperature seed decoder nextLayerData

  feedbackToken :: Signal dom Token
  feedbackToken = regEn 0 readyPulse tokenSample

  selectedToken :: Signal dom Token
  selectedToken = mux inputTokenValid inputToken feedbackToken

  -- Embedding lookup
  tokenEmbedding = Embedding.embedder vocabulary <$> selectedToken

  layerDataRegister :: Signal dom LayerData
  layerDataRegister = register initialLayerData nextLayerData

  layerInputSelector :: ProcessingState -> LayerData -> Vec ModelDimension FixedPoint -> LayerData
  layerInputSelector ps current tokenEmb
    | processingStage ps /= Stage1_ProjectQKV = current
    | processingLayer ps == 0                 = current { inputVector = tokenEmb }
    | otherwise                               = current { inputVector = feedForwardOutput current }

  selectedInput :: Signal dom LayerData
  selectedInput = liftA3 layerInputSelector processingState layerDataRegister tokenEmbedding

  (nextLayerData, doneFlags, kErrVec, vErrVec) =
      pipelineProcessorDbg processingState selectedInput cacheOwners transformerLayers

  (writeDone, attnDone) = unzip doneFlags

  writeDoneThisLayer = (!!) <$> sequenceA writeDone <*> layerIndex
  attnDoneThisLayer  = (!!) <$> sequenceA attnDone <*> layerIndex

--------------------------------------------------------------------------------
-- Debug pipeline processor
--------------------------------------------------------------------------------

type LayerProcessorDataDbg dom =
  (Signal dom LayerData, (Signal dom Bool, Signal dom Bool), Signal dom Bool, Signal dom Bool)

pipelineProcessorDbg :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom LayerData
  -> Vec NumLayers (Cache.KVRamOwner dom)
  -> Vec NumLayers TransformerLayer.TransformerLayerComponent
  -> ( Signal dom LayerData
     , Vec NumLayers (Signal dom Bool, Signal dom Bool)
     , Vec NumLayers (Signal dom Bool)
     , Vec NumLayers (Signal dom Bool)
     )
pipelineProcessorDbg processingState initialData cacheOwners transformerLayers =
  let
    indexedLayers =
      imap (\i (comp, owner) -> (i, comp, owner))
           (zip transformerLayers cacheOwners)

    -- result :: (Signal LayerData, Vec NumLayers ((Signal Bool, Signal Bool), Signal Bool, Signal Bool))
    (finalLayerData, results) =
      mapAccumL (layerProcessorDbg processingState) initialData indexedLayers

    (flags, kErrs, vErrs) = unzip3 results
  in
    (finalLayerData, flags, kErrs, vErrs)

layerProcessorDbg :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom ProcessingState
  -> Signal dom LayerData
  -> (Index NumLayers, TransformerLayer.TransformerLayerComponent, Cache.KVRamOwner dom)
  -> (Signal dom LayerData, ( (Signal dom Bool, Signal dom Bool)
                            , Signal dom Bool
                            , Signal dom Bool))
layerProcessorDbg processingState currentData (layerIx, layerComp, cacheOwner) =
  let
    (newData, wDone, attnDone, commitC3, kErr, vErr) =
      TLDbg.multiCycleTransformerLayerDbg layerComp cacheOwner layerIx processingState currentData

    selectedData =
      liftA4 (layerDataSelector layerIx) processingState currentData newData commitC3

  in
    (selectedData, ((wDone, attnDone), kErr, vErr))

layerDataSelector :: Index NumLayers
                  -> ProcessingState
                  -> LayerData -> LayerData -> LayerData -> LayerData
layerDataSelector lIx ps oldD newD c3D
  | processingLayer ps /= lIx    = oldD
  | processingStage ps == Stage3_Attend = c3D
  | otherwise                    = newD
