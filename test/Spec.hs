module Spec (main) where

import Clash.Prelude

import Test.Hspec
import qualified Model.Layers.Attention.AttentionHeadSpec (spec)
import qualified Model.Layers.Attention.MultiHeadAttentionSpec (spec)
import qualified Model.Layers.FeedForward.FeedForwardNetworkSpec  (spec)
import qualified Model.Layers.TransformerLayerSpec (spec)
import qualified Model.Helpers.MatVecI8ESpec (spec)
import qualified Model.Layers.TransformerLayer.ControlOneHeadSpec (spec)

main :: IO ()
main = hspec $ do
  Model.Layers.Attention.AttentionHeadSpec.spec
  Model.Layers.Attention.MultiHeadAttentionSpec.spec
  Model.Layers.FeedForward.FeedForwardNetworkSpec.spec
  Model.Layers.TransformerLayerSpec.spec
  Model.Layers.TransformerLayer.ControlOneHeadSpec.spec
  Model.Helpers.MatVecI8ESpec.spec
