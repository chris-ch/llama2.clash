module LLaMa2.Top
  ( topEntityWithAxi, Decoder.DecoderIntrospection(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.LayerData ( Temperature, Seed, Token )

import qualified LLaMa2.Decoder.Decoder as Decoder ( decoder, DecoderIntrospection(..) )
import LLaMa2.ParamsPlaceholder (decoderConst)
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn)
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut)

topEntityWithAxi
  :: HiddenClockResetEnable System
  => Signal System Bool                -- ^ bypassBoot (True for simulation)
  -> Slave.AxiSlaveIn System
  -> Slave.AxiSlaveIn System
  -> Signal System Bool                -- ^ powerOn
  -> Signal System Token
  -> Signal System Bool
  -> Signal System Temperature
  -> Signal System Seed
  -> ( Signal System Token
     , Signal System Bool
     , Master.AxiMasterOut System
     , Master.AxiMasterOut System
     , Signal System Bool
     , Signal System (Unsigned 32)
     , Decoder.DecoderIntrospection System
     )
topEntityWithAxi bypassBoot emmcSlave ddrSlave powerOn inTok inTokValid temp seed =
  Decoder.decoder bypassBoot emmcSlave ddrSlave powerOn decoderConst inTok inTokValid temp seed
