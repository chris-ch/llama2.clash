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
