module Spec (main) where

import Clash.Prelude

import Test.Hspec
import qualified LLaMa2.Layer.Attention.LayerWeightBufferSpec (spec)
import qualified LLaMa2.Layer.TransformerLayerSpec (spec)
import qualified LLaMa2.Numeric.OperationsSpec (spec)
import qualified LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec (spec)
import qualified Simulation.AxiWriteMasterSpec (spec)
import qualified Simulation.WeightLoaderSpec (spec)
import qualified LLaMa2.Decoder.MultiTokenSpec (spec)
import qualified LLaMa2.Decoder.TimingValidationSpec (spec)

main :: IO ()
main = hspec $ do
  LLaMa2.Layer.TransformerLayerSpec.spec
  LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec.spec
  LLaMa2.Numeric.OperationsSpec.spec
  Simulation.AxiWriteMasterSpec.spec
  Simulation.WeightLoaderSpec.spec
  LLaMa2.Layer.Attention.LayerWeightBufferSpec.spec
  --LLaMa2.Decoder.MultiTokenSpec.spec
  LLaMa2.Decoder.TimingValidationSpec.spec
