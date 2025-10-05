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

-- Helper to simulate N cycles and extract results
simulateNCycles ::
  Int ->
  [Bool] ->
  [Vec 4 FixedPoint] ->
  ([Vec 3 FixedPoint], [Bool], [Bool])
simulateNCycles n validIns vecs =
  let inputs = DL.zip validIns vecs
      inputSig = fromList inputs

      -- Expose clock, reset, enable to the function
      (outVecsSig, validOutsSig, readyOutsSig) =
        exposeClockResetEnable
          (sequentialMatVecStub makeSimpleQMatrix)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen
          inputSig

      outVecs = DL.take n $ sample outVecsSig
      validOuts = DL.take n $ sample validOutsSig
      readyOuts = DL.take n $ sample readyOutsSig
   in (outVecs, validOuts, readyOuts)

-- Helper to simulate N cycles and extract results
simulateNCycles' ::
  Int ->
  [Bool] ->
  [Vec 4 FixedPoint] ->
  ([Vec 3 FixedPoint], [Bool], [Bool])
simulateNCycles' n validIns vecs =
  let inputs = DL.zip validIns vecs
      inputSig = fromList inputs

      -- Expose clock, reset, enable to the function
      (outVecsSig, validOutsSig, readyOutsSig) =
        exposeClockResetEnable
          (sequentialMatVec makeSimpleQMatrix)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen
          inputSig

      outVecs = DL.take n $ sample outVecsSig
      validOuts = DL.take n $ sample validOutsSig
      readyOuts = DL.take n $ sample readyOutsSig
   in (outVecs, validOuts, readyOuts)

spec :: Spec
spec = do
  describe "sequentialMatVecStub" $ do
    it "initializes with readyOut=True and validOut=False" $ do
      let (_outVecs, validOuts, readyOuts) =
            simulateNCycles 1 [False] [repeat 0]
          readyOut0 = DL.head readyOuts
          validOut0 = DL.head validOuts
      readyOut0 `shouldBe` True
      validOut0 `shouldBe` False

    it "accepts input when validIn=True and readyOut=True" $ do
      let (_, _, readyOuts) =
            simulateNCycles
              3
              [False, True, False]
              [repeat 0, makeSimpleVec, repeat 0]
          ready1 = readyOuts P.!! 1
      ready1 `shouldBe` True

    it "produces validOut one cycle after validIn" $ do
      let (_, validOuts, _) =
            simulateNCycles
              4
              [False, True, False, False]
              [repeat 0, makeSimpleVec, repeat 0, repeat 0]
          valid0 = DL.head validOuts
          valid1 = validOuts P.!! 1
          valid2 = validOuts P.!! 2
          valid3 = validOuts P.!! 3

      valid0 `shouldBe` False
      valid1 `shouldBe` False
      valid2 `shouldBe` True
      valid3 `shouldBe` False

    it "computes correct matrix-vector product" $ do
      let (outVecs, _, _) =
            simulateNCycles
              3
              [False, True, False]
              [repeat 0, makeSimpleVec, repeat 0]
          outVec = outVecs P.!! 2 -- Result appears when validOut=True

          -- Expected: [1*1 + 2*0.5 + 3*0.25 + 4*0.125,
          --            5*1 + 6*0.5 + 7*0.25 + 8*0.125,
          --            9*1 + 10*0.5 + 11*0.25 + 12*0.125]
          --         = [3.25, 10.75, 18.25]
          expected = 3.25 :> 10.75 :> 18.25 :> Nil
          tolerance = 0.01
      -- Check each element is within tolerance
      let diffs = P.zipWith (\a b -> abs (a - b)) (toList outVec) (toList expected)
      all (< tolerance) diffs `shouldBe` True

    it "maintains readyOut behavior through state transitions" $ do
      let (_, _, readyOuts) = simulateNCycles 4
                      [False, True, False, False]
                      [repeat 0, makeSimpleVec, repeat 0, repeat 0]
          ready0 = DL.head readyOuts
          ready1 = readyOuts P.!! 1
          ready2 = readyOuts P.!! 2
          ready3 = readyOuts P.!! 3

      ready0 `shouldBe` True   -- Idle, ready to accept
      ready1 `shouldBe` True   -- Just accepted, combinational ready still True
      ready2 `shouldBe` False  -- Busy (state=True from previous cycle)
      ready3 `shouldBe` True   -- Back to idle (state=False)

    it "handles back-to-back transactions" $ do
      let vec1 = 1.0 :> 1.0 :> 1.0 :> 1.0 :> Nil
          vec2 = 2.0 :> 2.0 :> 2.0 :> 2.0 :> Nil
          (outVecs, validOuts, _) =
            simulateNCycles
              6
              [False, True, True, False, False, False]
              [repeat 0, vec1, vec2, repeat 0, repeat 0, repeat 0]

          -- First result at cycle 2
          out2 = outVecs P.!! 2
          valid2 = validOuts P.!! 2
          -- Second result at cycle 3
          out3 = outVecs P.!! 3
          valid3 = validOuts P.!! 3

      valid2 `shouldBe` True
      valid3 `shouldBe` True

      -- First output: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
      let expected1 = 10.0 :> 26.0 :> 42.0 :> Nil
          diffs1 = P.zipWith (\a b -> abs (a - b)) (toList out2) (toList expected1)
      all (< 0.01) diffs1 `shouldBe` True

      -- Second output: [2*(1+2+3+4), 2*(5+6+7+8), 2*(9+10+11+12)] = [20, 52, 84]
      let expected2 = 20.0 :> 52.0 :> 84.0 :> Nil
          diffs2 = P.zipWith (\a b -> abs (a - b)) (toList out3) (toList expected2)
      all (< 0.01) diffs2 `shouldBe` True

    it "ignores inputs when validIn=False" $ do
      let (_, validOuts, _) =
            simulateNCycles
              4
              [False, False, True, False]
              [repeat 0, makeSimpleVec, makeSimpleVec, repeat 0]

          valid1 = validOuts P.!! 1
          valid2 = validOuts P.!! 2
          valid3 = validOuts P.!! 3

      valid1 `shouldBe` False -- No validIn at cycle 0
      valid2 `shouldBe` False -- No validIn at cycle 1
      valid3 `shouldBe` True -- validIn at cycle 2, result at cycle 3
    it "latches output correctly" $ do
      let (outVecs, _, _) =
            simulateNCycles
              5
              [False, True, False, False, False]
              [repeat 0, makeSimpleVec, repeat 0, repeat 0, repeat 0]

          out2 = outVecs P.!! 2
          out3 = outVecs P.!! 3
          out4 = outVecs P.!! 4

      -- Output should be latched and remain stable
      let diffs23 = P.zipWith (\a b -> abs (a - b)) (toList out2) (toList out3)
          diffs34 = P.zipWith (\a b -> abs (a - b)) (toList out3) (toList out4)

      all (< 0.0001) diffs23 `shouldBe` True
      all (< 0.0001) diffs34 `shouldBe` True

  describe "sequentialMatVec" $ do
    it "initializes with readyOut=True and validOut=False" $ do
      let (_outVecs, validOuts, readyOuts) =
            simulateNCycles' 1 [False] [repeat 0]
          readyOut0 = DL.head readyOuts
          validOut0 = DL.head validOuts
      readyOut0 `shouldBe` True
      validOut0 `shouldBe` False

    it "accepts input when validIn=True and readyOut=True" $ do
      let (_, _, readyOuts) =
            simulateNCycles'
              3
              [False, True, False]
              [repeat 0, makeSimpleVec, repeat 0]
          ready1 = readyOuts P.!! 1
      ready1 `shouldBe` True

    it "produces validOut one cycle after validIn" $ do
      let (_, validOuts, _) =
            simulateNCycles'
              4
              [False, True, False, False]
              [repeat 0, makeSimpleVec, repeat 0, repeat 0]
          valid0 = DL.head validOuts
          valid1 = validOuts P.!! 1
          valid2 = validOuts P.!! 2
          valid3 = validOuts P.!! 3

      valid0 `shouldBe` False
      valid1 `shouldBe` False
      valid2 `shouldBe` True
      valid3 `shouldBe` False

    it "computes correct matrix-vector product" $ do
      let (outVecs, _, _) =
            simulateNCycles'
              3
              [False, True, False]
              [repeat 0, makeSimpleVec, repeat 0]
          outVec = outVecs P.!! 2 -- Result appears when validOut=True

          -- Expected: [1*1 + 2*0.5 + 3*0.25 + 4*0.125,
          --            5*1 + 6*0.5 + 7*0.25 + 8*0.125,
          --            9*1 + 10*0.5 + 11*0.25 + 12*0.125]
          --         = [3.25, 10.75, 18.25]
          expected = 3.25 :> 10.75 :> 18.25 :> Nil
          tolerance = 0.01
      -- Check each element is within tolerance
      let diffs = P.zipWith (\a b -> abs (a - b)) (toList outVec) (toList expected)
      all (< tolerance) diffs `shouldBe` True

    it "maintains readyOut behavior through state transitions" $ do
      let (_, _, readyOuts) = simulateNCycles' 4
                      [False, True, False, False]
                      [repeat 0, makeSimpleVec, repeat 0, repeat 0]
          ready0 = DL.head readyOuts
          ready1 = readyOuts P.!! 1
          ready2 = readyOuts P.!! 2
          ready3 = readyOuts P.!! 3

      ready0 `shouldBe` True   -- Idle, ready to accept
      ready1 `shouldBe` True   -- Just accepted, combinational ready still True
      ready2 `shouldBe` False  -- Busy (state=True from previous cycle)
      ready3 `shouldBe` True   -- Back to idle (state=False)

    it "handles back-to-back transactions" $ do
      let vec1 = 1.0 :> 1.0 :> 1.0 :> 1.0 :> Nil
          vec2 = 2.0 :> 2.0 :> 2.0 :> 2.0 :> Nil
          (outVecs, validOuts, _) =
            simulateNCycles'
              6
              [False, True, True, False, False, False]
              [repeat 0, vec1, vec2, repeat 0, repeat 0, repeat 0]

          -- First result at cycle 2
          out2 = outVecs P.!! 2
          valid2 = validOuts P.!! 2
          -- Second result at cycle 3
          out3 = outVecs P.!! 3
          valid3 = validOuts P.!! 3

      valid2 `shouldBe` True
      valid3 `shouldBe` True

      -- First output: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
      let expected1 = 10.0 :> 26.0 :> 42.0 :> Nil
          diffs1 = P.zipWith (\a b -> abs (a - b)) (toList out2) (toList expected1)
      all (< 0.01) diffs1 `shouldBe` True

      -- Second output: [2*(1+2+3+4), 2*(5+6+7+8), 2*(9+10+11+12)] = [20, 52, 84]
      let expected2 = 20.0 :> 52.0 :> 84.0 :> Nil
          diffs2 = P.zipWith (\a b -> abs (a - b)) (toList out3) (toList expected2)
      all (< 0.01) diffs2 `shouldBe` True

    it "ignores inputs when validIn=False" $ do
      let (_, validOuts, _) =
            simulateNCycles'
              4
              [False, False, True, False]
              [repeat 0, makeSimpleVec, makeSimpleVec, repeat 0]

          valid1 = validOuts P.!! 1
          valid2 = validOuts P.!! 2
          valid3 = validOuts P.!! 3

      valid1 `shouldBe` False -- No validIn at cycle 0
      valid2 `shouldBe` False -- No validIn at cycle 1
      valid3 `shouldBe` True -- validIn at cycle 2, result at cycle 3
    it "latches output correctly" $ do
      let (outVecs, _, _) =
            simulateNCycles'
              5
              [False, True, False, False, False]
              [repeat 0, makeSimpleVec, repeat 0, repeat 0, repeat 0]

          out2 = outVecs P.!! 2
          out3 = outVecs P.!! 3
          out4 = outVecs P.!! 4

      -- Output should be latched and remain stable
      let diffs23 = P.zipWith (\a b -> abs (a - b)) (toList out2) (toList out3)
          diffs34 = P.zipWith (\a b -> abs (a - b)) (toList out3) (toList out4)

      all (< 0.0001) diffs23 `shouldBe` True
      all (< 0.0001) diffs34 `shouldBe` True
