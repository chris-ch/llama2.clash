module Model.Top
  ( topEntity
  ) where

import Clash.Prelude

import Model.Core.Types
  ( Temperature, Seed, Token
  )
import Model.Config
  ( ModelDimension, NumLayers, SequenceLength
  )

import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)
import qualified Model.Core.Transformer as Transformer

-- ====== top with attention tap out ======
topEntity
  :: forall dom
   . HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom Token  -- Input token
  -> Signal dom Bool           -- ^ inputTokenValid: high when inputTokenSignal carries the prompt token (pos 0)
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token                -- sampled token
     , Signal dom Bool                 -- ready pulse (end of last FFN)
     )
topEntity decoder = Transformer.multiCycleTransformer decoder (repeat Cache.makeRamOwnerKV)
