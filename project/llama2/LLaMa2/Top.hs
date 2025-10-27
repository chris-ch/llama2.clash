module LLaMa2.Top
  ( topEntityWithAxi, Decoder.DecoderIntrospection(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.LayerData ( Temperature, Seed, Token )

import qualified LLaMa2.Decoder.Decoder as Decoder ( decoder, DecoderIntrospection(..) )
import Simulation.ParamsPlaceholder (decoderConst)
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn)
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut)

topEntityWithAxi
  :: HiddenClockResetEnable System
  =>  Slave.AxiSlaveIn System
  -> Signal System Bool                -- ^ powerOn
  -> Signal System Token
  -> Signal System Bool
  -> Signal System Temperature
  -> Signal System Seed
  -> ( Signal System Token
     , Signal System Bool
     , Master.AxiMasterOut System
     , Decoder.DecoderIntrospection System
     )
topEntityWithAxi ddrSlave powerOn inTok inTokValid temp seed =
  Decoder.decoder ddrSlave powerOn decoderConst inTok inTokValid temp seed
