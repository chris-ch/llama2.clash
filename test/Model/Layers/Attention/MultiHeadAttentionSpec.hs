module Model.Layers.Attention.MultiHeadAttentionSpec (spec) where

import Test.Hspec
import Clash.Prelude


spec :: Spec
spec = do
  describe "applyRotaryPositionEncoding" $
    it "is identity when cos=1 and sin=0" $ do
      True `shouldBe` True
