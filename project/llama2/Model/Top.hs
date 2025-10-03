module Model.Top
  ( topEntity, topEntitySim
  ) where

import Clash.Prelude

import Model.Core.Types ( Temperature, Seed, Token )

import Model.Core.Transformer as Transformer
import Model.Params.Decoder (decoderConst)
import qualified Model.Layers.TransformerLayer as TransformerLayer (TransformerDecoderComponent)

-- Monomorphic, fully applied top. Domain: System
{-# ANN topEntity
  (Synthesize
    { t_name   = "llama2_decoder"
    , t_inputs = [ PortName "clk"
                 , PortName "rst"
                 , PortName "en"
                 , PortName "in_token"
                 , PortName "in_token_valid"
                 , PortName "temperature"
                 , PortName "seed"
                 ]
    , t_output = PortProduct ""
                   [ PortName "out_token"
                   , PortName "ready_pulse"
                   ]
    }) #-}
topEntity
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System Token
  -> Signal System Bool
  -> Signal System Temperature
  -> Signal System Seed
  -> (Signal System Token, Signal System Bool)
topEntity clk rst en inTok inTokValid temp seed =
  withClockResetEnable clk rst en $
    Transformer.multiCycleTransformer decoderConst inTok inTokValid temp seed

topEntitySim :: HiddenClockResetEnable System
  => TransformerLayer.TransformerDecoderComponent
  -> Signal System Token  -- Input token
  -> Signal System Bool           -- ^ inputTokenValid: high when inputTokenSignal carries the prompt token (pos 0)
  -> Signal System Temperature
  -> Signal System Seed
  -> ( Signal System Token                -- sampled token
     , Signal System Bool                 -- ready pulse (end of last FFN)
     )
topEntitySim = Transformer.multiCycleTransformer
