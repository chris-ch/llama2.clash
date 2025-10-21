module LLaMa2.Layer.TransformerLayerSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint)
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Types.ModelConfig (ModelDimension, HeadDimension)
import LLaMa2.Layer.Attention.MultiHeadAttention (singleHeadController)

-- | Simple deterministic WO matrix for testing
makeSimpleWOMatrix :: MatI8E ModelDimension HeadDimension
makeSimpleWOMatrix = imap
      (\i _ ->
         ( imap (\j _ -> fromIntegral (i * headDim) + (fromIntegral j + 1) :: Signed 8)
                (repeat 0 :: Vec HeadDimension (Signed 8))
         , 0))
      (repeat (repeat 0 :: Vec HeadDimension Int, 0 :: Int) :: Vec ModelDimension (Vec HeadDimension Int, Int))
 where
   headDim = snatToNum (SNat @HeadDimension)

-- | Head output with decreasing values: [1.0, 0.5, 0.333..., ...]
makeSimpleHeadOutput :: Vec HeadDimension FixedPoint
makeSimpleHeadOutput = imap (\i _ -> 1.0 / fromIntegral (i+1)) (repeat (0 :: Int))

spec :: Spec
spec = do
    it "produces defined outputs during reset" $ do
      let headOut = makeSimpleHeadOutput
          headOutputs = fromList $ DL.repeat headOut :: Signal System (Vec HeadDimension FixedPoint)
          headDones = fromList $ DL.repeat False :: Signal System Bool
          (_, validOutsSig, readyOutsSig) =
            exposeClockResetEnable
              (singleHeadController headDones headOutputs makeSimpleWOMatrix)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
          validOuts = DL.take 5 $ sample @System validOutsSig
          readyOuts = DL.take 5 $ sample @System readyOutsSig
      all P.not validOuts `shouldBe` True
    it "headDones signal is well-defined" $ do
      let headDonesList = DL.take 15 $ DL.replicate 10 False P.++ DL.repeat True
      all (\x -> P.not x || x) headDonesList `shouldBe` True
