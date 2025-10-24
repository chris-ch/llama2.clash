module Spec (main) where

import Clash.Prelude

import Test.Hspec
import qualified LLaMa2.Layer.Attention.LayerWeightBufferSpec (spec)
import qualified LLaMa2.Layer.TransformerLayerSpec (spec)
import qualified LLaMa2.Numeric.OperationsSpec (spec)
import qualified LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec (spec)
import qualified LLaMa2.Memory.AxiWriteMasterSpec (spec)
import qualified LLaMa2.Memory.WeightLoaderSpec (spec)
import qualified LLaMa2.Layer.Attention.QKVProjectionIntegrationSpec (spec)
import qualified LLaMa2.Decoder.MultiTokenSpec (spec)

main :: IO ()
main = hspec $ do
  LLaMa2.Layer.TransformerLayerSpec.spec
  LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec.spec
  LLaMa2.Numeric.OperationsSpec.spec
  LLaMa2.Memory.AxiWriteMasterSpec.spec
  LLaMa2.Memory.WeightLoaderSpec.spec
  LLaMa2.Layer.Attention.LayerWeightBufferSpec.spec
  LLaMa2.Layer.Attention.QKVProjectionIntegrationSpec.spec
  LLaMa2.Decoder.MultiTokenSpec.spec
