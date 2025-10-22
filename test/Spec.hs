module Spec (main) where

import Clash.Prelude

import Test.Hspec
import qualified LLaMa2.Layer.Attention.AttentionHeadSpec (spec)
import qualified LLaMa2.Layer.Attention.MultiHeadAttentionSpec (spec)
import qualified LLaMa2.Layer.FeedForward.FeedForwardNetworkSpec  (spec)
import qualified LLaMa2.Layer.TransformerLayerSpec (spec)
import qualified LLaMa2.Helpers.MatVecI8ESpec (spec)
import qualified LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec (spec)
import qualified LLaMa2.Memory.AxiReadMasterSpec (spec)
import qualified LLaMa2.Memory.AxiWriteMasterSpec (spec)
import qualified LLaMa2.Memory.WeightLoaderSpec (spec)

main :: IO ()
main = hspec $ do
  LLaMa2.Layer.Attention.AttentionHeadSpec.spec
  LLaMa2.Layer.Attention.MultiHeadAttentionSpec.spec
  LLaMa2.Layer.FeedForward.FeedForwardNetworkSpec.spec
  LLaMa2.Layer.TransformerLayerSpec.spec
  LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec.spec
  LLaMa2.Helpers.MatVecI8ESpec.spec
  LLaMa2.Memory.AxiReadMasterSpec.spec
  LLaMa2.Memory.AxiWriteMasterSpec.spec
  LLaMa2.Memory.WeightLoaderSpec.spec