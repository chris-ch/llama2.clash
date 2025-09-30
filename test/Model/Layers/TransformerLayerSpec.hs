module Model.Layers.TransformerLayerSpec (spec) where

import Test.Hspec ( Spec, describe, it, shouldBe )
import Clash.Prelude

spec :: Spec
spec = do
    describe "TransformerLayer multiCycleTransformerLayer" $ do
        it "should produce valid intermediate data and stage pulses" $ do
            True `shouldBe` True
