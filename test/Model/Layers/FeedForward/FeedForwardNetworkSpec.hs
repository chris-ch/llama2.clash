module Model.Layers.FeedForward.FeedForwardNetworkSpec (spec) where

import Test.Hspec
import Clash.Prelude
import Model.Layers.FeedForward.FeedForwardNetwork.Internal
    ( FeedForwardNetworkComponent(FeedForwardNetworkComponent, fRMSFfn,
                                  fW1, fW2, fW3) )
import Model.Core.Types ( CArray2D(..), HiddenDim, ModelDim )
import Model.Layers.FeedForward.FeedForwardNetwork (computeFeedForward)

spec :: Spec
spec = describe "FeedForwardNetwork.computeFeedForward" $ do
  it "produces a vector with all elements equal" $ do
    let
      countRows1 :: Vec ModelDim Float
      countRows1 = generate (SNat @ModelDim) (+ 0.001) (0.0 :: Float)

      countRows2 :: Vec HiddenDim Float
      countRows2 = generate (SNat @HiddenDim) (+ 0.001) (0.0 :: Float)

      mat1 :: CArray2D HiddenDim ModelDim
      mat1 = CArray2D $ repeat countRows1

      mat2 :: CArray2D ModelDim HiddenDim
      mat2 = CArray2D $ repeat countRows2

      ffn = FeedForwardNetworkComponent { fW1 = mat1, fW2 = mat2, fW3 = mat1, fRMSFfn = repeat 1.0 }
      inputVec = repeat 0.5 :: Vec ModelDim Float
      outputVec = computeFeedForward ffn inputVec
    all (== head outputVec) outputVec `shouldBe` True
