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

topEntity :: HiddenClockResetEnable System
  => TransformerLayer.TransformerDecoderComponent
  -> Signal System Token  -- Input token
  -> Signal System Bool           -- ^ inputTokenValid: high when inputTokenSignal carries the prompt token (pos 0)
  -> Signal System Temperature
  -> Signal System Seed
  -> ( Signal System Token                -- sampled token
     , Signal System Bool                 -- ready pulse (end of last FFN)
     )
topEntity decoder = Transformer.multiCycleTransformer decoder (repeat Cache.makeRamOwnerKV)
