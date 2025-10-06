module Model.Helpers.MatVecI8ESpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import Model.Helpers.MatVecI8E
import Model.Numeric.ParamPack (QArray2D (..))
import Model.Numeric.Types (FixedPoint)
import Test.Hspec
import qualified Prelude as P

-- Helper to create a simple quantized matrix for testing
makeSimpleQMatrix :: QArray2D 3 4
makeSimpleQMatrix =
  QArray2D
    $ (1 :> 2 :> 3 :> 4 :> Nil, 0)
    :> (5 :> 6 :> 7 :> 8 :> Nil, 0)
    :> (9 :> 10 :> 11 :> 12 :> Nil, 0)
    :> Nil

-- Helper to create a simple input vector
makeSimpleVec :: Vec 4 FixedPoint
makeSimpleVec = 1.0 :> 0.5 :> 0.25 :> 0.125 :> Nil

-- Simulate until we get a valid output, return the result
-- This waits for the handshake to complete regardless of cycle count
simulateUntilValid ::
  Int -> -- Max cycles to simulate
  Vec 4 FixedPoint -> -- Input vector
  Maybe (Vec 3 FixedPoint) -- Result (Nothing if timeout)
simulateUntilValid maxCycles vec =
  let -- Create input stream: wait for ready, then send valid input
      inputStream = (False, repeat 0) : DL.repeat (True, vec)
      inputSig = fromList inputStream

      -- Run simulation
      (outVecsSig, validOutsSig, readyOutsSig) =
        exposeClockResetEnable
          (sequentialMatVecStub makeSimpleQMatrix)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen
          inputSig

      outVecs = DL.take maxCycles $ sample outVecsSig
      validOuts = DL.take maxCycles $ sample validOutsSig
      _readyOuts = DL.take maxCycles $ sample readyOutsSig

      -- Find first cycle where validOut is True
      resultsWithIndex = DL.zip3 [0 :: Int ..] validOuts outVecs
      validResults = DL.filter (\(_, valid, _) -> valid) resultsWithIndex
   in case validResults of
        [] -> Nothing -- Timeout: no valid output within maxCycles
        ((_, _, result) : _) -> Just result

-- Simulate until we get a valid output, return the result
-- This waits for the handshake to complete regardless of cycle count
simulateUntilValid' ::
  Int -> -- Max cycles to simulate
  Vec 4 FixedPoint -> -- Input vector
  Maybe (Vec 3 FixedPoint) -- Result (Nothing if timeout)
simulateUntilValid' maxCycles vec =
  let -- Create input stream: wait for ready, then send valid input
      inputStream = (False, repeat 0) : DL.repeat (True, vec)
      inputSig = fromList inputStream

      -- Run simulation
      (outVecsSig, validOutsSig, readyOutsSig) =
        exposeClockResetEnable
          (sequentialMatVec makeSimpleQMatrix)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen
          inputSig

      outVecs = DL.take maxCycles $ sample outVecsSig
      validOuts = DL.take maxCycles $ sample validOutsSig
      _readyOuts = DL.take maxCycles $ sample readyOutsSig

      -- Find first cycle where validOut is True
      resultsWithIndex = DL.zip3 [0 :: Int ..] validOuts outVecs
      validResults = DL.filter (\(_, valid, _) -> valid) resultsWithIndex

   in case validResults of
        [] -> Nothing -- Timeout: no valid output within maxCycles
        ((_, _, result) : _) -> Just result

-- Helper to check if result matches expected within tolerance
checkResultWithinTolerance :: Vec 3 FixedPoint -> Vec 3 FixedPoint -> FixedPoint -> Bool
checkResultWithinTolerance actual expected tolerance =
  let diffs = P.zipWith (\a b -> abs (a - b)) (toList actual) (toList expected)
   in all (< tolerance) diffs

spec :: Spec
spec = do
  describe "sequentialMatVecStub" $ do
    it "computes correct matrix-vector product (cycle-independent)" $ do
      let result = simulateUntilValid 10 makeSimpleVec

          -- Expected: [1*1 + 2*0.5 + 3*0.25 + 4*0.125,
          --            5*1 + 6*0.5 + 7*0.25 + 8*0.125,
          --            9*1 + 10*0.5 + 11*0.25 + 12*0.125]
          --         = [3.25, 10.75, 18.25]
          expected = 3.25 :> 10.75 :> 18.25 :> Nil
          tolerance = 0.01

      -- Check that we got a result
      result `shouldSatisfy` ( \case
                            Nothing -> False
                            Just _ -> True
                        )
      -- Check the result is correct
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output received within timeout"

    it "handles zero vector correctly" $ do
      let zeroVec = 0.0 :> 0.0 :> 0.0 :> 0.0 :> Nil
          result = simulateUntilValid 10 zeroVec
          expected = 0.0 :> 0.0 :> 0.0 :> Nil
          tolerance = 0.01

      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output received within timeout"

    it "handles all-ones vector correctly" $ do
      let onesVec = 1.0 :> 1.0 :> 1.0 :> 1.0 :> Nil
          result = simulateUntilValid 10 onesVec
          -- Expected: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
          expected = 10.0 :> 26.0 :> 42.0 :> Nil
          tolerance = 0.01

      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output received within timeout"

    it "handles negative values correctly" $ do
      let negVec = (-1.0) :> (-0.5) :> 0.5 :> 1.0 :> Nil
          result = simulateUntilValid 10 negVec
          -- Expected: [1*(-1) + 2*(-0.5) + 3*0.5 + 4*1,
          --            5*(-1) + 6*(-0.5) + 7*0.5 + 8*1,
          --            9*(-1) + 10*(-0.5) + 11*0.5 + 12*1]
          --         = [3.5, 3.5, 3.5]
          expected = 3.5 :> 3.5 :> 3.5 :> Nil
          tolerance = 0.01
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output received within timeout"

    it "handles unit basis vectors correctly (column extraction)" $ do
      -- Test with unit vector [1, 0, 0, 0] - should extract first column
      let unitVec1 = 1.0 :> 0.0 :> 0.0 :> 0.0 :> Nil
          result1 = simulateUntilValid 10 unitVec1
          expected1 = 1.0 :> 5.0 :> 9.0 :> Nil
          tolerance = 0.01

      case result1 of
        Just outVec ->
          checkResultWithinTolerance outVec expected1 tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output for unit vector [1,0,0,0]"

      -- Test with unit vector [0, 0, 0, 1] - should extract last column
      let unitVec4 = 0.0 :> 0.0 :> 0.0 :> 1.0 :> Nil
          result4 = simulateUntilValid 10 unitVec4
          expected4 = 4.0 :> 8.0 :> 12.0 :> Nil

      case result4 of
        Just outVec ->
          checkResultWithinTolerance outVec expected4 tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output for unit vector [0,0,0,1]"

    it "produces deterministic results for same input" $ do
      let vec = 0.7 :> 0.3 :> 0.2 :> 0.1 :> Nil
          result1 = simulateUntilValid 10 vec
          result2 = simulateUntilValid 10 vec

      case (result1, result2) of
        (Just out1, Just out2) -> do
          let matches =
                P.zipWith
                  (\a b -> abs (a - b) < 0.0001)
                  (toList out1)
                  (toList out2)
          DL.and matches `shouldBe` True
        _ ->
          expectationFailure "Failed to get results for determinism test"

    it "handles large magnitude values" $ do
      let largeVec = 10.0 :> 20.0 :> 30.0 :> 40.0 :> Nil
          result = simulateUntilValid 10 largeVec
          -- Expected: [1*10 + 2*20 + 3*30 + 4*40,
          --            5*10 + 6*20 + 7*30 + 8*40,
          --            9*10 + 10*20 + 11*30 + 12*40]
          --         = [300, 910, 1520]
          expected = 300.0 :> 700.0 :> 1100.0 :> Nil
          tolerance = 0.1

      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output received within timeout"

    it "respects ready/valid handshaking protocol" $ do
      -- Verify that readyOut goes low when busy and validOut appears
      let inputStream = (False, repeat 0) : (True, makeSimpleVec) : DL.repeat (False, repeat 0)
          inputSig = fromList inputStream

          (_, validOutsSig, readyOutsSig) =
            exposeClockResetEnable
              (sequentialMatVecStub makeSimpleQMatrix)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              inputSig

          validOuts = DL.take 10 $ sample validOutsSig
          readyOuts = DL.take 10 $ sample readyOutsSig

          -- Check that we eventually get a valid output
          hasValidOut = DL.or validOuts
          -- Check that ready signal changes (not stuck)
          readyChanges = P.length (DL.nub readyOuts) > 1

      hasValidOut `shouldBe` True
      readyChanges `shouldBe` True

  describe "sequentialMatVec" $ do
    
    it "computes correct matrix-vector product (cycle-independent)" $ do
      let result = simulateUntilValid' 60 makeSimpleVec

          -- Expected: [1*1 + 2*0.5 + 3*0.25 + 4*0.125,
          --            5*1 + 6*0.5 + 7*0.25 + 8*0.125,
          --            9*1 + 10*0.5 + 11*0.25 + 12*0.125]
          --         = [3.25, 10.75, 18.25]
          expected = 3.25 :> 10.75 :> 18.25 :> Nil
          tolerance = 0.01

      -- Check that we got a result
      result `shouldSatisfy` ( \case
                            Nothing -> False
                            Just _ -> True
                        )

      -- Check the result is correct
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output received within timeout"

    it "handles negative values correctly" $ do
      let 
          negVec = (-1.0) :> (-0.5) :> 0.5 :> 1.0 :> Nil
          result = simulateUntilValid' 60 negVec
          -- Expected: [1*(-1) + 2*(-0.5) + 3*0.5 + 4*1,
          --            5*(-1) + 6*(-0.5) + 7*0.5 + 8*1,
          --            9*(-1) + 10*(-0.5) + 11*0.5 + 12*1]
          --         = [3.5, 3.5, 3.5]
          expected = 3.5 :> 3.5 :> 3.5 :> Nil
          tolerance = 0.01

      -- Check that we got a result
      result `shouldSatisfy` ( \case
                            Nothing -> False
                            Just _ -> True
                        )

      -- Check the result is correct
      case result of
        Just outVec ->
          checkResultWithinTolerance outVec expected tolerance `shouldBe` True
        Nothing ->
          expectationFailure "No valid output received within timeout"

    it "produces deterministic results for same input" $ do
      let vec = 0.3 :> 0.2 :> 0.1 :> 0.0 :> Nil
          r1 = simulateUntilValid' 60 vec
          r2 = simulateUntilValid' 60 vec
      case (r1, r2) of
        (Just out1, Just out2) ->
          P.and (P.zipWith (\a b -> abs (a-b) < 0.0001) (toList out1) (toList out2)) `shouldBe` True
        _ -> expectationFailure "No results for determinism test"

    it "handles ready/valid protocol correctly for sequentialMatVec" $ do
      let inputStream = (False, repeat 0) : (True, makeSimpleVec) : DL.repeat (False, repeat 0)
          inputSig = fromList inputStream
          (_, validOutsSig, readyOutsSig) =
            exposeClockResetEnable
              (sequentialMatVec makeSimpleQMatrix)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              inputSig
          validOuts = DL.take 60 $ sample validOutsSig
          readyOuts = DL.take 60 $ sample readyOutsSig
      DL.or validOuts `shouldBe` True
      P.length (DL.nub readyOuts) > 1 `shouldBe` True
      