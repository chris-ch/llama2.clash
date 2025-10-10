module Spec (main) where

import Clash.Prelude

import Test.Hspec
import qualified LLaMa2.Layers.Attention.AttentionHeadSpec (spec)
import qualified LLaMa2.Layers.Attention.MultiHeadAttentionSpec (spec)
import qualified LLaMa2.Layers.FeedForward.FeedForwardNetworkSpec  (spec)
import qualified LLaMa2.Layers.TransformerLayerSpec (spec)
import qualified LLaMa2.Helpers.MatVecI8ESpec (spec)
import qualified LLaMa2.Layers.TransformerLayer.ControlOneHeadSpec (spec)

main :: IO ()
main = hspec $ do
  LLaMa2.Layers.Attention.AttentionHeadSpec.spec
  LLaMa2.Layers.Attention.MultiHeadAttentionSpec.spec
  LLaMa2.Layers.FeedForward.FeedForwardNetworkSpec.spec
  LLaMa2.Layers.TransformerLayerSpec.spec
  LLaMa2.Layers.TransformerLayer.ControlOneHeadSpec.spec
  LLaMa2.Helpers.MatVecI8ESpec.spec
