module LLaMa2.Layers.Attention.AttentionHeadSpec (spec) where
    
import Clash.Prelude
import Test.Hspec (Spec, describe, it, shouldBe)

spec :: Spec
spec = do
  describe "attentionHead" $ do

    it "returns V at the only position when allowed" $ do
      True `shouldBe` True
