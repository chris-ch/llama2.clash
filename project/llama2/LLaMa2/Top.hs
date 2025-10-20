module LLaMa2.Top
  ( topEntity, topEntitySim, Decoder.DecoderIntrospection(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.LayerData ( Temperature, Seed, Token )

import qualified LLaMa2.Decoder.Decoder as Decoder ( decoder, DecoderIntrospection(..) )
import LLaMa2.Params.Decoder (decoderConst)
import LLaMa2.Types.Parameters (DecoderParameters)

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
        , PortProduct "introspect"
            [ PortName "state"           -- ProcessingState (sum type encoded)
            , PortName "layerIndex"
            , PortName "ready"
            , PortName "logitsValid"
            , PortName "attnDone"
            , PortName "qkvDone"
            , PortName "ffnDone"
            , PortName "writeDone"
            , PortName "inputToken"
            , PortName "selectedToken"
            , PortName "feedbackToken"
            , PortName "embeddingNorm"
            , PortName "outputNorm"
            ]
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
  -> (
    Signal System Token
  , Signal System Bool
  , Decoder.DecoderIntrospection System -- ^ introspection signals
  )
topEntity clk rst en inTok inTokValid temp seed =
  withClockResetEnable clk rst en $
    Decoder.decoder decoderConst inTok inTokValid temp seed

topEntitySim :: HiddenClockResetEnable System
  => DecoderParameters
  -> Signal System Token  -- Input token
  -> Signal System Bool           -- ^ inputTokenValid: high when inputTokenSignal carries the prompt token (pos 0)
  -> Signal System Temperature
  -> Signal System Seed
  -> ( Signal System Token                -- sampled token
      , Signal System Bool                 -- ready pulse (end of last FFN)
      , Decoder.DecoderIntrospection System -- ^ introspection signals
     )
topEntitySim = Decoder.decoder
