module Model.Layers.TransformerLayerSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import Model.Numeric.ParamPack (QArray2D (..))
import Model.Numeric.Types (FixedPoint, Exponent)
import Test.Hspec
import qualified Prelude as P
import Model.Layers.TransformerLayer.Internal (controlOneHead)
import Model.Config (ModelDimension, HeadDimension)

-- | Simple deterministic WO matrix for testing
makeSimpleWOMatrix :: QArray2D ModelDimension HeadDimension
makeSimpleWOMatrix =
  QArray2D $
    imap
      (\i _ ->
         ( imap (\j _ -> fromIntegral (i * headDim) + (fromIntegral j + 1) :: Signed 8)
                (repeat 0 :: Vec HeadDimension (Signed 8))
         , 0))  -- exponent = 0
      (repeat (repeat 0 :: Vec HeadDimension Int, 0 :: Int) :: Vec ModelDimension (Vec HeadDimension Int, Int))
 where
   headDim = snatToNum (SNat @HeadDimension)

-- | Head output with decreasing values: [1.0, 0.5, 0.333..., ...]
makeSimpleHeadOutput :: Vec HeadDimension FixedPoint
makeSimpleHeadOutput = imap (\i _ -> 1.0 / fromIntegral (i+1)) (repeat (0 :: Int))

-- | Compute expected projection given WO and headOutput
expectedProjection ::
  QArray2D ModelDimension HeadDimension ->
  Vec HeadDimension FixedPoint ->
  Vec ModelDimension FixedPoint
expectedProjection (QArray2D rows) headOut =
  imap (\i _ -> dotProduct (rows !! i) headOut) (repeat (0 :: Int))
 where
   dotProduct :: (Vec HeadDimension (Signed 8), Exponent)
              -> Vec HeadDimension FixedPoint
              -> FixedPoint
   dotProduct (mantissas, expnt) inp =
     let scale = 2 ^^ expnt :: FixedPoint
         vecFP = map (\x -> fromIntegral x * scale) mantissas
     in sum (zipWith (*) vecFP inp)

-- Simulate until we get a valid output
simulateControlOneHeadUntilValid ::
  Int ->                                    -- Max cycles to simulate
  [(Vec HeadDimension FixedPoint, Bool)] -> -- Input stream: (headOutput, headDone)
  Maybe (Vec ModelDimension FixedPoint)     -- Result
simulateControlOneHeadUntilValid maxCycles inputList =
  let
      headOutputs = fromList $ P.map P.fst inputList
      headDones   = fromList $ P.map P.snd inputList

      (projOutsSig, validOutsSig, _readyOutsSig) =
        exposeClockResetEnable
          (controlOneHead headOutputs headDones makeSimpleWOMatrix)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen

      projOuts  = DL.take maxCycles $ sample projOutsSig
      validOuts = DL.take maxCycles $ sample validOutsSig
      resultsWithIndex = DL.zip3 [0 :: Int ..] validOuts projOuts
      validResults = DL.filter (\(_, valid, _) -> valid) resultsWithIndex
   in case validResults of
        [] -> Nothing
        ((_, _, result) : _) -> Just result

-- | Tolerance-based check
checkResultWithinTolerance ::
  Vec ModelDimension FixedPoint ->
  Vec ModelDimension FixedPoint ->
  FixedPoint -> Bool
checkResultWithinTolerance actual expected tolerance =
  let diffs = P.zipWith (\a b -> abs (a - b)) (toList actual) (toList expected)
   in all (< tolerance) diffs

-- | Helper to build an input stream with a single headDone pulse
createInputStreamWithPulse ::
  Int ->                                -- Cycle to pulse headDone
  Vec HeadDimension FixedPoint ->       -- Head output value
  Int ->                                -- Total length
  [(Vec HeadDimension FixedPoint, Bool)]
createInputStreamWithPulse pulseAt headOut totalLen =
  [ (headOut, i == pulseAt) | i <- [0 .. totalLen - 1] ]

spec :: Spec
spec = do
  describe "controlOneHead" $ do
    it "computes correct WO projection when headDone pulses" $ do
      let headOut    = makeSimpleHeadOutput
          inputStream = createInputStreamWithPulse 2 headOut 100
          result     = simulateControlOneHeadUntilValid 100 inputStream
          expected   = expectedProjection makeSimpleWOMatrix headOut
          tolerance  = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing -> expectationFailure "No valid output received"

{- 
    it "handles zero head output" $ do
      let zeroHeadOut = repeat 0 :: Vec HeadDimension FixedPoint
          inputStream = createInputStreamWithPulse 2 zeroHeadOut 100
          result     = simulateControlOneHeadUntilValid 100 inputStream
          expected   = expectedProjection makeSimpleWOMatrix zeroHeadOut
          tolerance  = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing -> expectationFailure "No valid output received"

    it "handles all-ones head output" $ do
      let onesHeadOut = repeat 1.0 :: Vec HeadDimension FixedPoint
          inputStream = createInputStreamWithPulse 2 onesHeadOut 100
          result     = simulateControlOneHeadUntilValid 100 inputStream
          expected   = expectedProjection makeSimpleWOMatrix onesHeadOut
          tolerance  = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing -> expectationFailure "No valid output received"

    it "handles negative head output values" $ do
      let negHeadOut = imap (\i _ -> if even i then -1 else 1) (repeat (0 :: Int))
          inputStream = createInputStreamWithPulse 2 negHeadOut 100
          result     = simulateControlOneHeadUntilValid 100 inputStream
          expected   = expectedProjection makeSimpleWOMatrix negHeadOut
          tolerance  = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing -> expectationFailure "No valid output received"

    it "produces deterministic results for same input" $ do
      let headOut    = imap (\i _ -> 0.1 * fromIntegral (i+1)) (repeat (0 :: Int))
          inputStream = createInputStreamWithPulse 2 headOut 100
          result1    = simulateControlOneHeadUntilValid 100 inputStream
          result2    = simulateControlOneHeadUntilValid 100 inputStream
      case (result1, result2) of
        (Just out1, Just out2) ->
          DL.and (P.zipWith (\a b -> abs (a - b) < 0.0001) (toList out1) (toList out2))
            `shouldBe` True
        _ -> expectationFailure "Failed to get deterministic results"

    it "respects ready/valid handshaking protocol" $ do
      let headOut    = makeSimpleHeadOutput
          inputStream = createInputStreamWithPulse 2 headOut 100
          headOutputs = fromList $ P.map P.fst inputStream
          headDones   = fromList $ P.map P.snd inputStream
          (_, validOutsSig, readyOutsSig) =
            exposeClockResetEnable
              (controlOneHead headOutputs headDones makeSimpleWOMatrix)
              CS.systemClockGen CS.resetGen CS.enableGen
          validOuts = DL.take 100 $ sample validOutsSig
          readyOuts = DL.take 100 $ sample readyOutsSig
      DL.or validOuts `shouldBe` True
      (P.length (DL.nub readyOuts) > 1) `shouldBe` True

    it "readyOut is True initially" $ do
      let headOut    = makeSimpleHeadOutput
          inputStream = createInputStreamWithPulse 2 headOut 100
          headOutputs = fromList $ P.map P.fst inputStream
          headDones   = fromList $ P.map P.snd inputStream
          (_, _, readyOutsSig) =
            exposeClockResetEnable
              (controlOneHead headOutputs headDones makeSimpleWOMatrix)
              CS.systemClockGen CS.resetGen CS.enableGen
          readyOuts = DL.take 100 $ sample readyOutsSig
      DL.head readyOuts `shouldBe` True

    it "handles headDone pulse at different cycles" $ do
      let headOut    = repeat 0.5 :: Vec HeadDimension FixedPoint
          inputStream = createInputStreamWithPulse 5 headOut 100
          result     = simulateControlOneHeadUntilValid 100 inputStream
          expected   = expectedProjection makeSimpleWOMatrix headOut
          tolerance  = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing -> expectationFailure "No valid output received"

    it "ignores continuous high headDone" $ do
      let headOut    = makeSimpleHeadOutput
          inputStream = [ (headOut, i >= 2) | i <- [(0 :: Int) .. 99] ]
          headOutputs = fromList $ P.map P.fst inputStream
          headDones   = fromList $ P.map P.snd inputStream
          (_, validOutsSig, _) =
            exposeClockResetEnable
              (controlOneHead headOutputs headDones makeSimpleWOMatrix)
              CS.systemClockGen CS.resetGen CS.enableGen
          validOuts  = DL.take 100 $ sample validOutsSig
          validCount = P.length $ P.filter id validOuts
      validCount `shouldBe` 1
 -}
{- 
    it "handles multiple sequential headDone pulses" $ do
      let headOut1 = replace 0 1.0 (repeat 0 :: Vec HeadDimension FixedPoint)
          headOut2 = replace 1 1.0 (repeat 0 :: Vec HeadDimension FixedPoint)
          inputStream =
            [(headOut1, False), (headOut1, False)] ++
            [(headOut1, True)] ++ DL.replicate 30 (headOut1, False) ++
            [(headOut2, True)] ++ DL.replicate 30 (headOut2, False)
          headOutputs = fromList $ P.map P.fst inputStream
          headDones   = fromList $ P.map P.snd inputStream
          (_, validOutsSig, _) =
            exposeClockResetEnable
              (controlOneHead headOutputs headDones makeSimpleWOMatrix)
              CS.systemClockGen CS.resetGen CS.enableGen
          validOuts  = DL.take 70 $ sample validOutsSig
          validCount = P.length $ P.filter id validOuts
      validCount `shouldBe` 2
 -}
