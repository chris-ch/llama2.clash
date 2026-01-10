module LLaMa2.Layer.Attention.QueryHeadProjector
  ( queryHeadProjector
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( SequenceLength,
      HeadDimension,
      NumQueryHeads,
      NumLayers,
      ModelDimension )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.QueryHeadCore as QueryHeadCore


--------------------------------------------------------------------------------
-- Top Level: queryHeadProjector
-- Wraps QueryHeadCore and applies rotary encoding
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- inputValid
  -> Signal dom Bool                              -- downStreamReady
  -> Signal dom Bool                              -- consumeSignal
  -> Signal dom (Index SequenceLength)            -- stepCount
  -> Signal dom (Vec ModelDimension FixedPoint)   -- xHat
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     , QueryHeadCore.QHeadDebugInfo dom
     )
queryHeadProjector cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal stepCount xHat params =
  ( QueryHeadCore.qhcAxiMaster core
  , qWithRotary
  , QueryHeadCore.qhcOutputValid core
  , QueryHeadCore.qhcReady core
  , QueryHeadCore.qhcDebug core
  )
  where
    core = QueryHeadCore.queryHeadCore cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat params

    -- Apply rotary encoding to output
    qWithRotary = (rotaryEncoder (PARAM.rotaryEncoding params) <$> stepCount) <*> QueryHeadCore.qhcResult core
