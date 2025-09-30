module Model.Core.Transformer.Internal where

import Clash.Prelude
import Helpers (liftA4)
import Model.Core.Types
  ( IntermediateData(..)
  , ProcessingState (..)
  , CycleStage (..)
  , NumLayers, Temperature, Seed
  , VocabularySize, Token, ModelDimemsion, SequenceLength
  )
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer
  ( TransformerLayerComponent(..)
  , TransformerDecoderComponent(..)
  )
import Model.Layers.TransformerLayer (TransformerDecoderComponent(..))
import Data.Maybe (isJust)
import qualified Model.Embedding.PRNG as PRNG
import qualified Model.Core.PipelineController as PipelineController
  ( runPipelineController
  , PipelineOutputs (..)
  )
import Model.Numeric.Types (FixedPoint)
import Model.Layers.Components.Quantized (EmbeddingComponentQ(..))
import qualified Model.Core.Embedding as Embedding (embed)

initialIntermediateData :: IntermediateData
initialIntermediateData = IntermediateData
  { inputVector       = repeat 0
  , queryVectors      = repeat (repeat 0)
  , keyVectors        = repeat (repeat 0)
  , valueVectors      = repeat (repeat 0)
  , attentionOutput   = repeat 0
  , feedForwardOutput = repeat 0
  }

outputTokenSignal
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom (Unsigned 32)
  -> TransformerLayer.TransformerDecoderComponent
  -> Signal dom IntermediateData
  -> Signal dom (Unsigned 32)
outputTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal =
  regEn 0 readyPulseSignal
        (PRNG.sampledTokenSignal readyPulseSignal temperatureSignal seedSignal decoder nextIntermediateDataSignal)

-- Embedding lookup using quantized table (I8E row -> FixedPoint vec)
embedFromComponent :: EmbeddingComponentQ -> Token -> Vec ModelDimemsion FixedPoint
embedFromComponent emb = Embedding.embed (vocabularyQ emb)
