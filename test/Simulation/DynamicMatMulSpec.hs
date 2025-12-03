module Simulation.DynamicMatMulSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P

import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.Operations (parallelRowMatrixMultiplierDyn)
import Simulation.Parameters (DecoderParameters(..), MultiHeadAttentionComponentQ(..), multiHeadAttention, QueryHeadComponentQ (..))
import qualified Simulation.ParamsPlaceholder as PARAM
import LLaMa2.Types.ModelConfig (HeadDimension, ModelDimension)
import LLaMa2.Numeric.Quantization (RowI8E (..))

-- Compute elementwise closeness
listsClose :: [FixedPoint] ->[FixedPoint] -> Bool
listsClose v1 v2 =
  P.and $ P.zipWith (\x y -> abs (realToFrac x - realToFrac y) < (0.1 :: Float))
                     v1 v2

-- Convert a row of (mantissa, exponent) to FixedPoint vector
rowToFixedPoint :: RowI8E n -> [FixedPoint]
rowToFixedPoint RowI8E { rowMantissas = mant, rowExponent = expon} = P.map (scalePow2F expon . fromIntegral) (toList mant)

spec :: Spec
spec =
  describe "Dynamic Row Matrix Multiplier" $ do
    it "produces output matching the constant Q-matrix for a known input" $ do
      -- Grab parameters
      let params :: DecoderParameters
          params = PARAM.decoderConst

          firstLayer = head (modelLayers params)
          mhaQ       = multiHeadAttention firstLayer
          firstHeadQ = head (qHeads mhaQ)
          qMatrix'    = qMatrix firstHeadQ

      -- Constant input vector
      let inputVec :: Vec ModelDimension FixedPoint
          inputVec = repeat 1

      let
        outDyn :: Signal System (Vec HeadDimension FixedPoint)
        (outDyn, _, _) = exposeClockResetEnable
                             (parallelRowMatrixMultiplierDyn
                               (pure True)
                               (pure True)
                               (pure qMatrix')
                               (pure inputVec))
                             systemClockGen resetGen enableGen

      -- Determine pipeline latency: 1 cycle + ceil(cols / 64)
      let pipelineLatency = 2  -- adjust based on your parallel64RowProcessor config

      -- Sample enough cycles
      let
          sampledOut = sampleN (pipelineLatency + 1) outDyn

      -- Take the first non-zero row
      let
        sampledRow :: [FixedPoint]
        sampledRow = toList $ sampledOut P.!! pipelineLatency

      -- Expected first row in FixedPoint
      let
        expectedRow :: [FixedPoint]
        expectedRow = rowToFixedPoint (head qMatrix')

      -- Assertion: the computed row should be elementwise close
      listsClose sampledRow expectedRow `shouldBe` True
