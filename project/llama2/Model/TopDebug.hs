module Model.TopDebug
  ( topEntityDebug
  ) where

import Clash.Prelude
import Model.Core.Types ( Temperature, Seed, Token )
import Model.Config ( NumLayers )
import qualified Model.Memory.KVCacheBank as Cache
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)
import qualified Model.Core.Transformer.Debug as TransformerDbg

topEntityDebug
  :: HiddenClockResetEnable dom
  => TransformerLayer.TransformerDecoderComponent
  -> Signal dom Token
  -> Signal dom Bool
  -> Signal dom Temperature
  -> Signal dom Seed
  -> ( Signal dom Token
     , Signal dom Bool
     , Signal dom (Vec NumLayers Bool)  -- K row error
     , Signal dom (Vec NumLayers Bool)  -- V row error
     )
topEntityDebug decoder tok valid temp seed =
  ( outTok
  , outReady
  , bundle kErrs   -- ✅ convert Vec (Signal Bool) -> Signal (Vec Bool)
  , bundle vErrs
  )
 where
  (outTok, outReady, kErrs, vErrs) =
    TransformerDbg.multiCycleTransformerDebug decoder (repeat Cache.makeRamOwnerKV) tok valid temp seed
