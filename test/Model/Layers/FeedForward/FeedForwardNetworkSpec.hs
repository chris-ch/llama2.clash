module Model.Layers.FeedForward.FeedForwardNetworkSpec (spec) where

import Test.Hspec
import Clash.Prelude
import Model.Layers.FeedForward.FeedForwardNetwork.Internal
    ( FeedForwardNetworkComponent(FeedForwardNetworkComponent, fRMSFfn,
                                  fW1, fW2, fW3) )
import Model.Core.Types ( CArray2D(..), HiddenDimension, ModelDimemsion )
import Model.Layers.FeedForward.FeedForwardNetwork (computeFeedForward)

spec :: Spec
spec = describe "FeedForwardNetwork.computeFeedForward" $ do
  it "produces a vector with all elements equal" $ do
    let
      countRows1 :: Vec ModelDimemsion Float
      countRows1 = generate (SNat @ModelDimemsion) (+ 0.001) (0.0 :: Float)

      countRows2 :: Vec HiddenDimension Float
      countRows2 = generate (SNat @HiddenDimension) (+ 0.001) (0.0 :: Float)

      mat1 :: CArray2D HiddenDimension ModelDimemsion
      mat1 = CArray2D $ repeat countRows1

      mat2 :: CArray2D ModelDimemsion HiddenDimension
      mat2 = CArray2D $ repeat countRows2

      ffn = FeedForwardNetworkComponent { fW1 = mat1, fW2 = mat2, fW3 = mat1, fRMSFfn = repeat 1.0 }
      inputVec = repeat 0.5 :: Vec ModelDimemsion Float
      outputVec = computeFeedForward ffn inputVec
    all (== head outputVec) outputVec `shouldBe` True
