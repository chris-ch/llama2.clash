module Model.Helpers.MatVecI8ESpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import Model.Config (HeadDimension, ModelDimension)
import Model.Helpers.MatVecI8E
import Model.Numeric.ParamPack (QArray2D (..), RowI8E)
import Model.Numeric.Types (Exponent, FixedPoint, Mantissa)
import Test.Hspec
import qualified Prelude as P

-- ==========
-- Simulation Helpers
-- ==========

-- Deterministic WO matrix: each output row uses mantissas [1..HeadDimension], exponent 0
makeWO :: QArray2D ModelDimension HeadDimension
makeWO =
  let rowMant :: Vec HeadDimension Mantissa
      rowMant = map (fromIntegral . (1 +) . fromEnum) (indicesI @HeadDimension)
      oneRow :: (Vec HeadDimension Mantissa, Exponent)
      oneRow = (rowMant, 0)
      rows :: Vec ModelDimension (Vec HeadDimension Mantissa, Exponent)
      rows = repeat oneRow
   in QArray2D rows

-- Simulate sequentialMatVec with a single valid pulse
simulateSingleValid ::
  Int -> -- Max cycles to simulate
  Int -> -- Cycle to fire valid (>=1 recommended)
  Vec HeadDimension FixedPoint -> -- Input vector
  ( [Vec ModelDimension FixedPoint], -- Output vectors
    [Bool], -- Valid outputs
    [Bool] -- Ready outputs
  )
simulateSingleValid maxCycles fireAt vec =
  let
    enableStream = P.replicate fireAt False P.++ [True] P.++ DL.repeat False
    enableSig = fromList enableStream
    inputStream = P.replicate fireAt (repeat 0) P.++ [vec] P.++ DL.repeat (repeat 0)
    inputSig = fromList inputStream
    (outVecsSig, validOutsSig, readyOutsSig, _, _, _) =
      exposeClockResetEnable
        (matrixMultiplier makeWO)
        CS.systemClockGen
        CS.resetGen
        CS.enableGen
        enableSig
        inputSig
   in ( DL.take maxCycles (sample outVecsSig),
        DL.take maxCycles (sample validOutsSig),
        DL.take maxCycles (sample readyOutsSig)
      )

-- Deterministic head vector: x_j = 1/(j+1)
makeHeadVec :: Vec HeadDimension FixedPoint
makeHeadVec =
  map
    (\j -> realToFrac (1.0 / (1.0 + fromIntegral (fromEnum j) :: Double)))
    (indicesI @HeadDimension)

-- Collect valid output events
collectValidEvents ::
  [Vec ModelDimension FixedPoint] ->
  [Bool] ->
  [(Int, Vec ModelDimension FixedPoint)]
collectValidEvents outs valids =
  [(i, o) | (i, (v, o)) <- P.zip [0 ..] (P.zip valids outs), v]

-- Golden projected vector using combinational kernel
goldenWOx :: Vec ModelDimension FixedPoint
goldenWOx = matrixVectorMult makeWO makeHeadVec

-- Helper to check if result matches expected within tolerance
withinTolVec :: FixedPoint -> Vec ModelDimension FixedPoint -> Vec ModelDimension FixedPoint -> Bool
withinTolVec tol a b =
  let diffs = P.zipWith (\x y -> abs (x - y)) (toList a) (toList b)
   in P.all (< tol) diffs

-- Conservative latency bound
worstLatency :: Int
worstLatency =
  let rows = natToNum @ModelDimension :: Int
      cols = natToNum @HeadDimension :: Int
   in rows * cols + rows + cols + 16

-- Helper to simulate rowStateMachine
simulateRowStateMachine :: Int -> [Bool] -> [Bool] -> [(Bool, Index 3, Bool)]
simulateRowStateMachine maxCycles validIns rowDones =
  let validInSig = fromList (validIns P.++ DL.repeat False) :: Signal System Bool
      rowDoneSig = fromList (rowDones P.++ DL.repeat False) :: Signal System Bool
      (busySig, rowIdxSig, clearRowSig) =
        exposeClockResetEnable
          (rowStateMachine @System @3 validInSig rowDoneSig)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen
      busys = sample busySig
      rowIdxs = sample rowIdxSig
      clearRows = sample clearRowSig
   in DL.take maxCycles $ P.zip3 busys rowIdxs clearRows

spec :: Spec
spec = do
  describe "accumulator" $ do
    it "correctly accumulates, holds, and resets" $ do
      let maxCycles = 10
          -- each reset is a warm-up period
          resetStream = [True, False, False, True, False, False, False, True, False, False, False]
          enableStream = [True, True, True, False, True, True, False, True, True, True, False]
          inputStream = [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0] :: [FixedPoint]
          expected = [0.0, 0.0, 1.0, 3.0, 0.0, 3.0, 7.0, 7.0, 5.0, 11.0, 18.0] :: [FixedPoint]
          enable = fromList enableStream :: Signal System Bool
          reset = fromList resetStream :: Signal System Bool
          input = fromList inputStream :: Signal System FixedPoint
          accSig =
            exposeClockResetEnable
              (accumulator reset enable input)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
          outputs = P.take maxCycles $ sample accSig
          tolerance = 0.01
      P.all (\(o, e) -> abs (o - e) < tolerance) (P.zip outputs expected) `shouldBe` True

  describe "columnComponentCounter" $ do
    it "correctly increments, resets, and holds the counter" $ do
      let maxCycles = 11
          resetStream = [True, False, False, False, True, False, False, True, False, False, False]
          enableStream = [False, True, True, True, False, True, False, True, True, True, False]
          expected = [0, 0, 1, 2, 3, 0, 1, 1, 0, 1, 2]
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              (columnComponentCounter @System @4 reset enable)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
          outputs = P.take maxCycles $ sample counter :: [Index 4]
          outputInts = P.map fromEnum outputs
      outputInts `shouldBe` expected
      let maxBoundVal = fromEnum (maxBound :: Index 4)
      all (<= maxBoundVal) outputInts `shouldBe` True

    it "ignores enable when reset" $ do
      let maxCycles = 11
          resetStream = [True, False, False, False, True, False, False, True, False, False, False]
          enableStream = [True, True, True, True, True, True, False, True, True, True, False]
          expected = [0, 0, 1, 2, 3, 0, 1, 1, 0, 1, 2]
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              (columnComponentCounter @System @4 reset enable)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
          outputs = P.take maxCycles $ sample counter :: [Index 4]
          outputInts = P.map fromEnum outputs

      outputInts `shouldBe` expected
      let maxBoundVal = fromEnum (maxBound :: Index 4)
      all (<= maxBoundVal) outputInts `shouldBe` True

  describe "singleRowProcessor" $ do
    context "computes dot product for a single row" $ do
      let maxCycles = 12

          rowVector :: RowI8E 4
          rowVector = (1 :> 2 :> 3 :> 4 :> Nil, 0)

          columnVector :: Vec 4 FixedPoint
          columnVector = 1.0 :> 0.5 :> 0.25 :> 0.125 :> Nil

          expected = 3.25 :: FixedPoint
          tolerance = 0.01

          -- Input signals
          row :: Signal System (RowI8E 4)
          row = pure rowVector

          reset :: Signal System Bool
          reset = fromList (True : P.replicate (maxCycles - 1) False)

          enable :: Signal dom Bool
          enable =
            fromList
              $ [False]
              P.++ DL.replicate 4 True
              P.++ DL.repeat False

          -- make sure to pad first cycle for reset warm-up
          column :: Signal dom (Vec 4 FixedPoint)
          column =
            fromList $ DL.replicate 5 columnVector
              P.++ DL.repeat (pure 0)

          -- Run simulation
          (outputComponent, rowDone) =
            exposeClockResetEnable
              singleRowProcessor
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
              row
              column

          outs = P.take maxCycles $ sample outputComponent
          dones = P.take maxCycles $ sample rowDone

          doneIndices = DL.findIndices id dones
          finalOut = if null doneIndices then 0 else outs P.!! P.last doneIndices

      it "there should be only one completion" $ do
        P.length doneIndices `shouldBe` 1
      it "final output should match expected value" $ do
        abs (finalOut - expected) < tolerance `shouldBe` True

    context "computes dot products for two rows sequentially" $ do
      let maxCycles = 20

          -- Define two different rows
          rowVector1 :: RowI8E 4
          rowVector1 = (1 :> 2 :> 3 :> 4 :> Nil, 0) -- First row: [1, 2, 3, 4], exponent 0
          rowVector2 :: RowI8E 4
          rowVector2 = (2 :> 3 :> 4 :> 5 :> Nil, 0) -- Second row: [2, 3, 4, 5], exponent 0
          columnVector :: Vec 4 FixedPoint
          columnVector = 1.0 :> 0.5 :> 0.25 :> 0.125 :> Nil -- Column: [1.0, 0.5, 0.25, 0.125]

          -- Expected dot products
          -- First row: 1*1.0 + 2*0.5 + 3*0.25 + 4*0.125 = 1.0 + 1.0 + 0.75 + 0.5 = 3.25
          expected1 = 3.25 :: FixedPoint
          -- Second row: 2*1.0 + 3*0.5 + 4*0.25 + 5*0.125 = 2.0 + 1.5 + 1.0 + 0.625 = 5.125
          expected2 = 5.125 :: FixedPoint

          tolerance = 0.01

          -- Input signals
          -- Process first row for 4 cycles, gap with reset, then second row for 4 cycles
          rowStream =
            [ rowVector1,
              rowVector1,
              rowVector1,
              rowVector1, -- First row
              rowVector1, -- Gap cycle (reset)
              rowVector2,
              rowVector2,
              rowVector2,
              rowVector2 -- Second row
            ]
              P.++ P.replicate (maxCycles - 9) rowVector2
          row :: Signal System (RowI8E 4)
          row = fromList rowStream

          resetStream =
            [True] -- Initial reset
              P.++ P.replicate 4 False -- First row processing
              P.++ [True] -- Reset after first row
              P.++ P.replicate (maxCycles - 6) False -- Second row and beyond
          reset :: Signal System Bool
          reset = fromList resetStream

          enableStream =
            [False] -- Initial idle cycle
              P.++ DL.replicate 4 True -- Process first row
              P.++ [False] -- Gap cycle
              P.++ DL.replicate 4 True -- Process second row
              P.++ DL.replicate (maxCycles - 10) False
          enable :: Signal System Bool
          enable = fromList enableStream

          columnStream =
            [pure 0] -- Initial idle cycle
              P.++ DL.replicate 4 columnVector -- First row columns
              P.++ [pure 0] -- Gap cycle
              P.++ DL.replicate 4 columnVector -- Second row columns
              P.++ DL.replicate (maxCycles - 10) (pure 0)
          column :: Signal System (Vec 4 FixedPoint)
          column = fromList columnStream

          -- Run simulation
          (outputComponent, rowDone) =
            exposeClockResetEnable
              singleRowProcessor
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
              row
              column

          outs = P.take maxCycles $ sample outputComponent
          dones = P.take maxCycles $ sample rowDone

          -- Find cycles where rowDone is True
          doneIndices = DL.findIndices id dones

          -- Extract outputs at done cycles
          finalOuts = [outs P.!! i | i <- doneIndices]

      it "has 2 completions" $ do
        putStrLn $ "finalOuts: " P.++ show finalOuts
        putStrLn $ "expected1: " P.++ show expected1
        putStrLn $ "expected2: " P.++ show expected2
        putStrLn $ "doneIndices: " P.++ show doneIndices
        P.length doneIndices `shouldBe` 2 -- Expect two row completions
      it "first result matches" $ do
        abs (DL.head finalOuts - expected1) < tolerance `shouldBe` True -- First row result
      it "second result matches" $ do
        abs (finalOuts P.!! 1 - expected2) < tolerance `shouldBe` True -- Second row result
      it "completions happen on cycles 5 and 10" $ do
        DL.head doneIndices `shouldBe` 5
        doneIndices P.!! 1 `shouldBe` 10

  describe "matrixMultiplier" $ do
    it "computes correct matrix-vector product for a small 2x3 example" $ do
      let tol = 0.01
          testMat =
            QArray2D
              ( (1 :> 2 :> 3 :> Nil, 0)
                  :> (4 :> 5 :> 6 :> Nil, 0)
                  :> Nil
              )
          xVec = 1 :> 2 :> 3 :> Nil
          expected = 14 :> 32 :> Nil
          -- Enough cycles to compute both rows sequentially
          maxCycles = 32

          -- Keep validIn True for number-of-columns per row
          enableStream =
            False -- idle cycle
              : True
              : True
              : True -- Complete first row (3 columns)
              : True
              : True
              : True -- Complete second row (3 columns)
              : DL.replicate (maxCycles - 7) False
          inputStream =
            repeat 0 -- idle cycle
              : xVec
              : xVec
              : xVec -- Complete first row (3 columns)
              : xVec
              : xVec
              : xVec -- Complete second row (3 columns)
              : DL.replicate (maxCycles - 7) (repeat 0)
          enableSig = fromList enableStream
          inputSig = fromList inputStream

          (outSig, validSig, readySig, yOutRow, doneCurrentRow, stateSignal) =
            exposeClockResetEnable
              (matrixMultiplier testMat)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              enableSig
              inputSig

          outs = DL.take maxCycles $ sample outSig
          valids = DL.take maxCycles $ sample validSig
          readys = DL.take maxCycles $ sample readySig
          validEvents = [(i, o) | (i, (v, o)) <- P.zip [0 :: Int ..] (P.zip valids outs), v]

      let (_, resultVec) = P.last validEvents
      putStrLn $ "doneCurrentRow: " P.++ show (DL.take maxCycles $ sample doneCurrentRow)
      putStrLn $ "yOutRow: " P.++ show (DL.take maxCycles $ sample yOutRow)
      putStrLn $ "state: " P.++ show (DL.take maxCycles $ sample stateSignal)
      putStrLn $ "outs: " P.++ show outs
      putStrLn $ "readys: " P.++ show readys
      putStrLn $ "Expected: " P.++ show expected

      let checkWithinTol a b tolerance = P.and $ P.zipWith (\x y -> abs (x - y) <= tolerance) (toList a) (toList b)

      validEvents `shouldSatisfy` (not . null)
      checkWithinTol resultVec expected tol `shouldBe` True

    it "computes correct matrix-vector product (sequentially, handshake-driven)" $ do
      let maxCycles = worstLatency
          tol = 0.01
          xVec = makeHeadVec
          -- Simulate with a single valid pulse
          (outs, valids, readys) = simulateSingleValid maxCycles 2 xVec
          validEvents = collectValidEvents outs valids

      let (_, resultVec) = P.last validEvents

      -- Expected result is the combinational matrix-vector product
      let expected = goldenWOx
      -- There should be at least one valid output
      validEvents `shouldSatisfy` (not . null)

      withinTolVec tol resultVec expected `shouldBe` True
      -- Ready should eventually become true again
      P.last readys `shouldBe` True

  -- Comprehensive test suite for rowStateMachine
  -- Output format: (busy, rowIdx, clearRow)

  describe "rowStateMachine - Core Behavior" $ do
    it "resets on first cycle even when idle" $ do
      let maxCycles = 4
          validIns = P.replicate maxCycles False
          rowDones = P.replicate maxCycles False
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected = [(False, 0, True), (False, 0, True), (False, 0, True), (False, 0, True)]
      outputs `shouldBe` expected

    it "starts processing with clearRow pulse" $ do
      let maxCycles = 4
          validIns = [False, True, False, False]
          rowDones = [False, False, False, False]
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected =
            [ (False, 0, True), -- Cycle 0: Initial reset
              (True, 0, True), -- Cycle 1: Start processing row 0, pulse clear
              (True, 0, False), -- Cycle 2-3: Processing row 0, no clear
              (True, 0, False)
            ]
      outputs `shouldBe` expected

    it "transitions to next row with clearRow pulse" $ do
      let maxCycles = 6
          validIns = [False, True, False, False, False, False]
          rowDones = [False, False, True, False, False, False]
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected =
            [ (False, 0, True), -- Cycle 0: Initial reset
              (True, 0, True), -- Cycle 1: Start row 0
              (True, 0, False), -- Cycle 2: Processing row 0
              (True, 1, True), -- Cycle 3: Row 0 done, move to row 1, pulse clear
              (True, 1, False), -- Cycle 4-5: Processing row 1
              (True, 1, False)
            ]
      outputs `shouldBe` expected

    it "returns to idle after completing all rows with clearRow pulse" $ do
      let maxCycles = 8
          validIns = [False, True, False, False, False, False, False, False]
          rowDones = [False, False, True, False, True, False, True, False]
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected =
            [ (False, 0, True), -- Cycle 0: Initial reset
              (True, 0, True), -- Cycle 1: Start row 0
              (True, 0, False), -- Cycle 2: Processing row 0
              (True, 1, True), -- Cycle 3: Move to row 1, pulse clear
              (True, 1, False), -- Cycle 4: Processing row 1
              (True, 2, True), -- Cycle 5: Move to row 2 (last), pulse clear
              (True, 2, False), -- Cycle 6: Processing row 2
              (False, 0, True) -- Cycle 7: Done, return to idle, pulse clear
            ]
      outputs `shouldBe` expected

    it "ignores validIn when already busy" $ do
      let maxCycles = 5
          validIns = [False, True, True, True, False] -- Extra True's should be ignored
          rowDones = [False, False, False, False, False]
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected =
            [ (False, 0, True), -- Cycle 0: Initial reset
              (True, 0, True), -- Cycle 1: Start row 0
              (True, 0, False), -- Cycle 2-4: Stay on row 0, ignore extra validIn
              (True, 0, False),
              (True, 0, False)
            ]
      outputs `shouldBe` expected

    it "requires rowDone to advance, not just time passing" $ do
      let maxCycles = 6
          validIns = [False, True, False, False, False, False]
          rowDones = [False, False, False, False, False, False] -- Never done
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected =
            [ (False, 0, True), -- Cycle 0: Initial reset
              (True, 0, True), -- Cycle 1: Start row 0
              (True, 0, False), -- Cycle 2-5: Stuck on row 0, waiting for rowDone
              (True, 0, False),
              (True, 0, False),
              (True, 0, False)
            ]
      outputs `shouldBe` expected

  describe "rowStateMachine - Edge Cases" $ do
    it "handles back-to-back operations" $ do
      let maxCycles = 10
          -- First operation: 3 rows
          -- Second operation: starts immediately after first completes
          validIns = [False, True, False, False, False, False, False, True, False, False]
          rowDones = [False, False, True, False, True, False, True, False, False, False]
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected =
            [ (False, 0, True), -- Cycle 0: Initial reset
              (True, 0, True), -- Cycle 1: Start first operation
              (True, 0, False), -- Cycle 2
              (True, 1, True), -- Cycle 3: Row 1
              (True, 1, False), -- Cycle 4
              (True, 2, True), -- Cycle 5: Row 2
              (True, 2, False), -- Cycle 6
              (False, 0, True), -- Cycle 7: First operation done, idle with clear
              (True, 0, True), -- Cycle 8: Start second operation
              (True, 0, False) -- Cycle 9
            ]
      outputs `shouldBe` expected

    it "clearRow only pulses for one cycle during transitions" $ do
      let maxCycles = 7
          validIns = [False, True, False, False, False, False, False]
          rowDones = [False, False, True, True, False, False, False] -- Multiple done pulses
          outputs = simulateRowStateMachine maxCycles validIns rowDones
          expected =
            [ (False, 0, True), -- Cycle 0: Initial reset
              (True, 0, True), -- Cycle 1: Start row 0, clear pulse
              (True, 0, False), -- Cycle 2: Processing
              (True, 1, True), -- Cycle 3: Advance to row 1, clear pulse (one cycle only)
              (True, 2, True), -- Cycle 4: Advance to row 2, clear pulse
              (True, 2, False), -- Cycle 5-6: On row 2, no clear
              (True, 2, False)
            ]
      outputs `shouldBe` expected
