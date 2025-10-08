module Model.Helpers.MatVecI8ESpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import Control.Monad.RWS (MonadState (put))
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
  let enableStream = P.replicate fireAt False P.++ [True] P.++ DL.repeat False
      enableSig = fromList enableStream
      inputStream = P.replicate fireAt (repeat 0) P.++ [vec] P.++ DL.repeat (repeat 0)
      inputSig = fromList inputStream
      (outVecsSig, validOutsSig, readyOutsSig, _, _, _, _, _, _, _, _, _) =
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
withinTolVec :: FixedPoint -> Vec size FixedPoint -> Vec size FixedPoint -> Bool
withinTolVec tol a b =
  let diffs = P.zipWith (\x y -> abs (x - y)) (toList a) (toList b)
   in P.all (< tol) diffs

-- Conservative latency bound
worstLatency :: Int
worstLatency =
  let rows = natToNum @ModelDimension :: Int
      cols = natToNum @HeadDimension :: Int
   in rows * cols + rows + cols + 16

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
          (outputComponent, rowDone, _acc, _) =
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
          (outputComponent, rowDone, _acc, _) =
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
        P.length doneIndices `shouldBe` 2 -- Expect two row completions
      it "first result matches" $ do
        abs (DL.head finalOuts - expected1) < tolerance `shouldBe` True -- First row result
      it "second result matches" $ do
        abs (finalOuts P.!! 1 - expected2) < tolerance `shouldBe` True -- Second row result
      it "completions happen on cycles 5 and 10" $ do
        DL.head doneIndices `shouldBe` 5
        doneIndices P.!! 1 `shouldBe` 10

    context "computes dot product for a 3-column row" $ do
      let maxCycles = 7
          rowVector :: RowI8E 3
          rowVector = (1 :> 2 :> 3 :> Nil, 0)
          columnVector :: Vec 3 FixedPoint
          columnVector = 1.0 :> 2.0 :> 3.0 :> Nil
          expected = 14.0 :: FixedPoint -- 1*1.0 + 2*2.0 + 3*3.0
          tolerance = 0.01
          -- Mimic matrixMultiplier signals
          resetStream = [False, True, False, False, False, True, False] -- FIX adjust timing (extra reset?)
          enableStream = [False, False, True, True, True, False, False] -- FIX adjust timing
          -- matrixMultiplier signal
          --enableStream = [False, False, True, True, True, True, False]
          row = pure rowVector
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          column = fromList $ pure 0 : pure 0 : P.replicate maxCycles columnVector  -- FIX requires padding
          -- Run simulation
          (outputComponent, rowDone, acc, colIdx) =
            exposeClockResetEnable
              singleRowProcessor
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
              row
              column
          colIndices = P.take maxCycles $ sample colIdx
          outs = P.take maxCycles $ sample outputComponent
          dones = P.take maxCycles $ sample rowDone
          accs = P.take maxCycles $ sample acc
          doneIndices = DL.findIndices id dones
          finalOut = if null doneIndices then 0 else outs P.!! P.last doneIndices
      it "completes after 3 columns (cycle 5)" $ do
        putStrLn $ "colIndices=" P.++ show colIndices
        putStrLn $ "accs=" P.++ show accs
        putStrLn $ "dones=" P.++ show dones
        putStrLn $ "doneIndices=" P.++ show doneIndices
        putStrLn $ "outs=" P.++ show outs
        doneIndices `shouldBe` [5]
      it "produces correct dot product" $ do
        abs (finalOut - expected) < tolerance `shouldBe` True
      it "accumulator follows expected sequence" $ do
        let expectedAccs = [0.0, 0.0, 1.0, 5.0, 14.0, 14.0, 0.0]
        P.all (\(a, e) -> abs (a - e) < tolerance) (P.zip (P.take maxCycles accs) expectedAccs) `shouldBe` True
      it "prints debug signals" $ do
        putStrLn $ "outs=" P.++ show outs
        putStrLn $ "dones=" P.++ show dones
        putStrLn $ "accs=" P.++ show accs

  describe "matrixMultiplierStateMachine" $ do
    context "correctly handles state transitions and control signals" $ do
      let -- Each tuple is (enable, allRowsDone)
          inputs =
            [ (False, False), -- Cycle 0: Stay in IDLE
              (True, False), -- Cycle 1: Transition to PROCESSING at next rising edge
              (False, False), -- Cycle 2: Stay in PROCESSING
              (False, False), -- Cycle 3: Stay in PROCESSING
              (False, True), -- Cycle 4: Transition to DONE
              (False, False), -- Cycle 5: Transition back to IDLE
              (False, False), -- Cycle 6: Stay in IDLE
              (False, False) -- Cycle 7: Stay in IDLE
            ]
          -- Each tuple is (expectedState, validOut, readyOut)
          expected =
            [ (IDLE, False, True), -- Cycle 0: Initial state
              (IDLE, False, True), -- Cycle 1: receiving enable=True
              (PROCESSING, False, False), -- Cycle 2: After enable=True
              (PROCESSING, False, False), -- Cycle 3: Continue PROCESSING
              (PROCESSING, False, False), -- Cycle 4: Continue PROCESSING
              (DONE, True, False), -- Cycle 5: After allRowsDone=True
              (IDLE, False, True), -- Cycle 6: After DONE
              (IDLE, False, True) -- Cycle 7: Stay in IDLE
            ]
          -- Split inputs into separate enable and allRowsDone signals
          enableStream = P.map fst inputs :: [Bool]
          allRowsDoneStream = P.map snd inputs :: [Bool]
          enableSig = fromList enableStream :: Signal System Bool
          allRowsDoneSig = fromList allRowsDoneStream :: Signal System Bool
          -- Simulate the state machine
          (stateSig, validSig, readySig) =
            exposeClockResetEnable
              matrixMultiplierStateMachine
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              enableSig
              allRowsDoneSig
          outputs =
            P.zip3
              (sample @System stateSig)
              (sample @System validSig)
              (sample @System readySig)
      it "computes outputs correctly" $ do
        let outputResults = P.take (P.length expected) outputs
            expectedResults = P.take (P.length expected) expected
        outputResults `shouldBe` expectedResults

  describe "matrixMultiplier" $ do
    context "computes matrix-vector multiplication for a 2x3 matrix" $ do
      let maxCycles = 20

          -- Test matrix: 2 rows x 3 columns
          -- Row 0: [1, 2, 3] with exponent 0
          -- Row 1: [4, 5, 6] with exponent 0
          testMat :: QArray2D 2 3
          testMat =
            QArray2D
              ( (1 :> 2 :> 3 :> Nil, 0)
                  :> (4 :> 5 :> 6 :> Nil, 0)
                  :> Nil
              )

          -- Input vector: [1, 2, 3]
          xVec :: Vec 3 FixedPoint
          xVec = 1 :> 2 :> 3 :> Nil

          -- Expected output: [14, 32]
          -- Row 0: 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
          -- Row 1: 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
          expected :: Vec 2 FixedPoint
          expected = 14 :> 32 :> Nil

          tolerance = 0.01

          -- Input signals
          -- Fire enable at cycle 1, hold input vector constant
          enableStream = [False, True] P.++ P.replicate (maxCycles - 2) False
          enable :: Signal System Bool
          enable = fromList enableStream

          inputStream = P.replicate maxCycles xVec
          inputVec :: Signal System (Vec 3 FixedPoint)
          inputVec = fromList inputStream

          -- Run simulation
          (outputVec, validOut, readyOut, innerState, rowResult, rowDone, rowIndex, acc, resetRow, enableRow, currentRow, colIdx) =
            exposeClockResetEnable
              (matrixMultiplier testMat)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              enable
              inputVec

          colIndices = P.take maxCycles $ sample colIdx
          currentRows = P.take maxCycles $ sample currentRow
          enableRows = P.take maxCycles $ sample enableRow
          resetRows = P.take maxCycles $ sample resetRow
          accs = P.take maxCycles $ sample acc
          rowIndices = P.take maxCycles $ sample rowIndex
          states = P.take maxCycles $ sample innerState
          rowResults = P.take maxCycles $ sample rowResult
          rowDones = P.take maxCycles $ sample rowDone
          outs = P.take maxCycles $ sample outputVec
          valids = P.take maxCycles $ sample validOut
          readys = P.take maxCycles $ sample readyOut

          -- Find cycles where valid is True and collect outputs
          validIndices = DL.findIndices id valids
          validEvents = [(i, outs P.!! i) | i <- validIndices]

      it "produces exactly one valid output" $ do
        putStrLn $ "colIndices=" P.++ show colIndices
        putStrLn $ "currentRows=" P.++ show currentRows
        putStrLn $ "enableRows=" P.++ show enableRows
        putStrLn $ "resetRows=" P.++ show resetRows
        putStrLn $ "accs=" P.++ show accs
        putStrLn $ "rowIndices=" P.++ show rowIndices
        putStrLn $ "rowResults=" P.++ show rowResults
        putStrLn $ "rowDones=" P.++ show rowDones
        putStrLn $ "states=" P.++ show states
        putStrLn $ "enable=" P.++ show (P.take maxCycles $ sample enable)
        putStrLn $ "input=" P.++ show xVec
        putStrLn $ "outs=" P.++ show outs
        putStrLn $ "readys=" P.++ show readys
        putStrLn $ "Valid events: " P.++ show validEvents
        P.length validEvents `shouldBe` 1

      it "output matches expected result" $ do
        let (_, result) = DL.head validEvents
        putStrLn $ "Result: " P.++ show result
        putStrLn $ "expected: " P.++ show expected
        withinTolVec tolerance result expected `shouldBe` True

      it "is ready initially (cycle 0)" $ do
        DL.head readys `shouldBe` True

      it "becomes not ready after accepting input" $ do
        -- After enable fires at cycle 1, ready should go low
        readys P.!! 2 `shouldBe` False

      it "returns to ready state after completion" $ do
        -- Find when valid fires and check ready is True in following cycles
        let (validCycle, _) = DL.head validEvents
            cycleAfterValid = validCycle + 1
        if cycleAfterValid < maxCycles
          then readys P.!! cycleAfterValid `shouldBe` True
          else pendingWith "Valid cycle too late to check ready state"

      it "valid output appears within reasonable latency" $ do
        let (validCycle, _) = DL.head validEvents
            -- For 2 rows x 3 cols, expect completion in roughly 2*3 + margin cycles
            maxExpectedLatency = 10
        validCycle `shouldSatisfy` (< maxExpectedLatency)
