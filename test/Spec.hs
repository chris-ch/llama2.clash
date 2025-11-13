module Spec (main) where

import Clash.Prelude

import Test.Hspec
import qualified LLaMa2.Layer.TransformerLayerSpec (spec)
import qualified LLaMa2.Numeric.OperationsSpec (spec)
import qualified LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec (spec)
import qualified Simulation.AxiWriteMasterSpec (spec)
import qualified Simulation.DRAMBackedAxiSlaveSpec  (spec)
import qualified Simulation.DynamicMatMulSpec (spec)


main :: IO ()
main = hspec $ do
  LLaMa2.Layer.TransformerLayerSpec.spec
  LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec.spec
  LLaMa2.Numeric.OperationsSpec.spec
  Simulation.AxiWriteMasterSpec.spec
  --LLaMa2.Decoder.MultiTokenSpec.spec
  Simulation.DRAMBackedAxiSlaveSpec.spec
  -- too long:
  --Simulation.WeightLoadingDiagnosticSpec.spec
  Simulation.DynamicMatMulSpec.spec

