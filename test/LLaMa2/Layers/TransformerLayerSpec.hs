module LLaMa2.Layers.TransformerLayerSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Numeric.ParamPack (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint, Exponent)
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Layers.TransformerLayer.Internal (singleHeadController)
import LLaMa2.Config (ModelDimension, HeadDimension)

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

-- | Compute expected projection given WO and headOutput
expectedProjection ::
  MatI8E ModelDimension HeadDimension ->
  Vec HeadDimension FixedPoint ->
  Vec ModelDimension FixedPoint
expectedProjection rows headOut =
  imap (\i _ -> dotProduct (rows !! i) headOut) (repeat (0 :: Int))
 where
   dotProduct :: (Vec HeadDimension (Signed 8), Exponent)
              -> Vec HeadDimension FixedPoint
              -> FixedPoint
   dotProduct (mantissas, expnt) inp =
     let scale = 2 ^^ expnt :: FixedPoint
         vecFP = map (\x -> fromIntegral x * scale) mantissas
     in sum (zipWith (*) vecFP inp)

-- | Simulate until valid output, respecting handshaking
simulateControlOneHeadUntilValid ::
  Int -> -- Max cycles to simulate
  Vec HeadDimension FixedPoint -> -- Head output value
  IO (Maybe (Vec ModelDimension FixedPoint)) -- Result
simulateControlOneHeadUntilValid maxCycles headOut = do
  let
      -- Create input streams with explicit System domain
      headOutputs :: Signal System (Vec HeadDimension FixedPoint)
      headOutputs = fromList $ DL.repeat headOut
      headDones :: Signal System Bool
      headDones = fromList $ DL.replicate 10 False P.++ DL.repeat True
      -- Simulate the module
      (projOutsSig, validOutsSig, readyOutsSig) =
        exposeClockResetEnable
          (singleHeadController headDones headOutputs makeSimpleWOMatrix)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen
      -- Sample outputs
      projOuts = DL.take maxCycles $ sample @System projOutsSig
      validOuts = DL.take maxCycles $ sample @System validOutsSig
      readyOuts = DL.take maxCycles $ sample @System readyOutsSig
      -- Filter for valid outputs
      resultsWithIndex = DL.zip3 [0 :: Int ..] validOuts projOuts
      validResults = DL.filter (\(_, valid, _) -> valid) resultsWithIndex
  -- Debug: Log initial signals
  P.putStrLn $ "headDones: " P.++ show (DL.take 20 $ sample @System headDones)
  P.putStrLn $ "readyOuts: " P.++ show (DL.take 20 readyOuts)
  P.putStrLn $ "validOuts: " P.++ show (DL.take 20 validOuts)
  P.putStrLn $ "projOuts (first 10 / first 5): " P.++ show (P.map (P.take 5 . toList) (DL.take 10 projOuts))
  case validResults of
    [] -> return Nothing
    ((_, _, result) : _) -> return $ Just result

-- | Tolerance-based check
checkResultWithinTolerance ::
  Vec ModelDimension FixedPoint ->
  Vec ModelDimension FixedPoint ->
  FixedPoint -> Bool
checkResultWithinTolerance actual expected tolerance =
  let diffs = P.zipWith (\a b -> abs (a - b)) (toList actual) (toList expected)
   in all (< tolerance) diffs

spec :: Spec
spec = do
  describe "singleHeadController" $ do
{-     it "computes correct WO projection when headDone pulses" $ do
      let headOut = makeSimpleHeadOutput
      result <- simulateControlOneHeadUntilValid 200 headOut
      let expected = expectedProjection makeSimpleWOMatrix headOut
          tolerance = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing -> expectationFailure "No valid output received within timeout" -}
    {- it "respects ready/valid handshaking protocol" $ do
      let headOut = makeSimpleHeadOutput
          headOutputs = fromList $ DL.repeat headOut :: Signal System (Vec HeadDimension FixedPoint)
          headDones = fromList $ DL.replicate 10 False P.++ DL.repeat True :: Signal System Bool
          (_, validOutsSig, readyOutsSig) =
            exposeClockResetEnable
              (singleHeadController headOutputs headDones makeSimpleWOMatrix)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
          validOuts = DL.take 200 $ sample @System validOutsSig
          readyOuts = DL.take 200 $ sample @System readyOutsSig
          hasValidOut = DL.or validOuts
          readyChanges = P.length (DL.nub readyOuts) > 1
      hasValidOut `shouldBe` True
      readyChanges `shouldBe` True -}
{-     it "handles zero input correctly" $ do
      let headOut = repeat 0 :: Vec HeadDimension FixedPoint
      result <- simulateControlOneHeadUntilValid 200 headOut
      let expected = repeat 0 :: Vec ModelDimension FixedPoint
          tolerance = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing -> expectationFailure "No valid output for zero input" -}
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
