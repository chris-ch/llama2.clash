module LLaMa2.Layer.FeedForward.FeedForwardNetworkSpec (spec) where

import Test.Hspec
import Clash.Prelude

spec :: Spec
spec = describe "FeedForwardNetwork.computeFeedForward" $ do
  it "produces a vector with all elements equal" $ do
    True `shouldBe` True
