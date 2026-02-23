module LLaMa2.Layer.Attention.KeyValueHeadProjector
  ( keyValueHeadProjector
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, SequenceLength )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS

--------------------------------------------------------------------------------
-- KV head projector
--
-- MIGRATION STATUS: K and V matrices are currently loaded from hardwired
-- (HC) parameters via 'pure (PARAM.kMatrix ...)' / 'pure (PARAM.vMatrix ...)'.
-- This is the same pattern used for Q *before* its DRAM migration.
--
-- DRAM migration plan for K (and V):
--   1. Replace 'pure (PARAM.kMatrix kvHeadParams)' with a call to
--      'kWeightLoader' from "LLaMa2.Layer.Attention.WeightLoader", which
--      already contains the AXI-fetching infrastructure and a DRAM-backed
--      address calculator.
--   2. Thread the AXI slave input and AXI master output through this
--      function and up through 'qkvProjector'.
--   3. In 'qkvProjector', wire KV heads into the AXI arbiter (currently
--      only Q heads are arbitrated) and pass 'consumeSignal' instead of
--      'downStreamReady' to KV heads, to coordinate multi-head clearing.
--   4. Add DRAM/HC mismatch assertions analogous to those in
--      'QueryHeadProjector' for regression testing during the transition.
--
-- Until the migration is complete, 'keyValueHeadProjector' has no AXI
-- master output and takes no DRAM slave input.
--------------------------------------------------------------------------------
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.KeyValueHeadComponentQ
  -> PARAM.RotaryEncodingComponentF
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     )
keyValueHeadProjector inputValid downStreamReady stepCountSig xHatSig kvHeadParams rotary =
  (kRoOut, vOut, outputValid, readyForInput)
 where
  selectedK :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedK = pure (PARAM.kMatrix kvHeadParams)

  selectedV :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedV = pure (PARAM.vMatrix kvHeadParams)

  (kOut, kValidOut, kReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedV xHatSig

  kRoOut = (rotaryEncoder rotary <$> stepCountSig) <*> kOut

  outputValid = kValidOut .&&. vValidOut
  readyForInput = kReadyOut .&&. vReadyOut
