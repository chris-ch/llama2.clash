module LLaMa2.Helpers.MatVecI8ESpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Helpers.MatVecI8E (accumulator, cyclicalCounter, singleRowProcessor, MultiplierState (..), matrixMultiplierStateMachine, matrixMultiplier)
import LLaMa2.Numeric.ParamPack (MatI8E, RowI8E)
import LLaMa2.Numeric.Types (FixedPoint)
import Test.Hspec
import qualified Prelude as P

-- ==========
-- Simulation Helpers
-- ==========

spec :: Spec
spec = do
  describe "accumulator" $ do
    context "correctly accumulates, holds, and resets" $ do
      let maxCycles = 10
          -- each reset is a warm-up period
          resetStream = [True, False, False, True, False, False, False, True, False, False, False]
          enableStream = [True, True, True, False, True, True, False, True, True, True, False]
          inputStream = [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0] :: [FixedPoint]
          expected = [0.0, 0.0, 1.0, 3.0, 0.0, 3.0, 7.0, 7.0, 0.0, 6.0] :: [FixedPoint]
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
      it "output stream as expected" $ do
        P.all (\(o, e) -> abs (o - e) < tolerance) (P.zip outputs expected) `shouldBe` True

    context "correctly accumulates with initial values" $ do
      let maxCycles = 10
          -- each reset is a warm-up period
          resetStream = [True, False, False, True, False, False, False, True, False, False, False]
          enableStream = [True, True, True, False, True, True, False, True, True, True, False]
          inputStream = [1.0, 2.0, 3.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0] :: [FixedPoint]
          expected = [0.0, 0.0, 2.0, 5.0, 0.0, 3.0, 7.0, 7.0, 0.0, 6.0] :: [FixedPoint]
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
      it "output stream as expected" $ do
        P.all (\(o, e) -> abs (o - e) < tolerance) (P.zip outputs expected) `shouldBe` True

  describe "columnComponentCounter" $ do
    context "correctly increments, resets, and holds the counter" $ do
      let maxCycles = 11
          resetStream = [True, False, False, False, True, False, False, True, False, False, False]
          enableStream = [False, True, True, True, False, True, False, True, True, True, False]
          expected = [0, 0, 1, 2, 3, 0, 1, 1, 0, 1, 2]
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              cyclicalCounter
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
          outputs = P.take maxCycles $ sample counter :: [Index 4]
          outputInts = P.map fromEnum outputs
      it "output stream as expected" $ do
        outputInts `shouldBe` expected
        let maxBoundVal = fromEnum (maxBound :: Index 4)
        all (<= maxBoundVal) outputInts `shouldBe` True

    context "ignores enable when reset" $ do
      let maxCycles = 11
          resetStream = [True, False, False, False, True, False, False, True, False, False, False]
          enableStream = [True, True, True, True, True, True, False, True, True, True, False]
          expected = [0, 0, 1, 2, 3, 0, 1, 1, 0, 1, 2]
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              cyclicalCounter
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
          outputs = P.take maxCycles $ sample counter :: [Index 4]
          outputInts = P.map fromEnum outputs
      it "output stream as expected" $ do
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
      let maxCycles = 12
          -- Define two different rows
          rowVector1 :: RowI8E 4
          rowVector1 = (1 :> 2 :> 3 :> 4 :> Nil, 0) -- First row: [1, 2, 3, 4], exponent 0
          rowVector2 :: RowI8E 4
          rowVector2 = (2 :> 3 :> 4 :> 5 :> Nil, 0) -- Second row: [2, 3, 4, 5], exponent 0
          columnVector :: Vec 4 FixedPoint
          columnVector = 1.0 :> 0.5 :> 0.25 :> 0.125 :> Nil -- Column: [1.0, 0.5, 0.25, 0.125]
          expectedOutsPattern =
            [ 0.0, -- Cycle 0: reset
              0.0, -- Cycle 1: start first row
              1.0, -- Cycle 2: 1*1.0
              2.0, -- Cycle 3: +2*0.5
              2.75, -- Cycle 4: +3*0.25
              3.25, -- Cycle 5: +4*0.125 (first row done)
              0.0, -- Cycle 6: reset before second row
              2.0, -- Cycle 7: start second row +2*1.0
              3.5, -- Cycle 8: +3*0.5
              4.5, -- Cycle 9: +4*0.25
              5.125, -- Cycle 10: + 5*0.125 (done)
              5.125 -- Cycle 11: still done
            ] ::
              [FixedPoint]
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
              P.++ [True] -- Gap cycle
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

      it "matches expected output" $ do
        outs `shouldBe` expectedOutsPattern
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
          resetStream = [False, True, False, False, False, False, False]
          enableStream = [False, False, True, True, True, False, False]
          row = pure rowVector
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          column = fromList $ pure 0 : pure 0 : P.replicate maxCycles columnVector -- padding with zero before enable
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
      it "completes after 3 columns (cycle 5)" $ do
        doneIndices `shouldBe` [5]
      it "produces correct dot product" $ do
        abs (finalOut - expected) < tolerance `shouldBe` True
      it "accumulator follows expected sequence" $ do
        let expectedOuts = [0.0, 0.0, 0.0, 1.0, 5.0, 14.0, 14.0]
        P.all (\(a, e) -> abs (a - e) < tolerance) (P.zip (P.take maxCycles outs) expectedOuts) `shouldBe` True

    context "computes dot product for a 3-column row, independently from components before enable" $ do
      let maxCycles = 7
          rowVector :: RowI8E 3
          rowVector = (1 :> 2 :> 3 :> Nil, 0)
          columnVector :: Vec 3 FixedPoint
          columnVector = 1.0 :> 2.0 :> 3.0 :> Nil
          expected = 14.0 :: FixedPoint -- 1*1.0 + 2*2.0 + 3*3.0
          tolerance = 0.01
          -- Mimic matrixMultiplier signals
          resetStream = [False, True, False, False, False, False, False]
          enableStream = [False, False, True, True, True, False, False]
          row = pure rowVector
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          column = pure columnVector
          
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
      it "completes after 3 columns (cycle 5)" $ do
        doneIndices `shouldBe` [5]
      it "produces correct dot product" $ do
        abs (finalOut - expected) < tolerance `shouldBe` True
      it "accumulator follows expected sequence" $ do
        let expectedOuts = [0.0, 0.0, 0.0, 1.0, 5.0, 14.0, 14.0]
        P.all (\(a, e) -> abs (a - e) < tolerance) (P.zip (P.take maxCycles outs) expectedOuts) `shouldBe` True

  describe "matrixMultiplierStateMachine" $ do
    context "correctly handles state transitions and control signals (assuming 3 rows and 4 columns)" $ do
      let maxCycles = 20
          -- Simulate processing: each row takes (numCols + 1) cycles
          -- Cycle 0: idle
          -- Cycle 1: enable goes high, transition to MReset
          -- Cycle 2: MReset (rowReset=True)
          -- Cycle 3-6: MProcessing (rowEnable=True), row 0
          -- Cycle 7: rowDone for row 0, transition to MReset
          -- Cycle 8: MReset (rowReset=True)
          -- Cycle 9-12: MProcessing, row 1
          -- Cycle 13: rowDone for row 1, transition to MReset
          -- Cycle 14: MReset (rowReset=True)
          -- Cycle 15-18: MProcessing, row 2
          -- Cycle 19: rowDone for row 2, transition to MDone

          -- Input: enable signal (high on cycle 1)
          enableStream =
            [False, True] P.++ P.replicate (maxCycles - 2) False
          enable :: Signal System Bool
          enable = fromList enableStream

          -- Input: rowDone signal (pulses after each row completes)
          -- Row processing takes numCols cycles, done appears 1 cycle after last enable
          rowDoneStream =
            P.replicate 7 False
              P.++ [True] -- Row 0 done at cycle 7
              P.++ P.replicate 5 False
              P.++ [True] -- Row 1 done at cycle 13
              P.++ P.replicate 5 False
              P.++ [True] -- Row 2 done at cycle 19
          rowDone :: Signal System Bool
          rowDone = fromList rowDoneStream

          -- Input: currentRow counter (0, 1, 2 cycling through rows)
          currentRowStream =
            P.replicate 8 0 -- Row 0 until cycle 7
              P.++ P.replicate 6 1 -- Row 1 until cycle 13
              P.++ P.replicate 6 2 -- Row 2 until cycle 19
          currentRow :: Signal System (Index 3)
          currentRow = fromList currentRowStream

          readyIn :: Signal System Bool
          readyIn = pure True

          (state, rowReset, rowEnable, validOut, readyOut) =
            exposeClockResetEnable
              matrixMultiplierStateMachine
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              enable
              readyIn
              rowDone
              currentRow

          states = P.take maxCycles $ sample state
          rowResets = P.take maxCycles $ sample rowReset
          rowEnables = P.take maxCycles $ sample rowEnable
          validOuts = P.take (maxCycles + 1) $ sample validOut
          readyOuts = P.take (maxCycles + 2) $ sample readyOut

          expectedStates =
            [ MIdle, -- Cycle 0: initial idle
              MIdle, -- Cycle 1: enable received, but state updates next cycle
              MReset, -- Cycle 2: reset row 0
              MProcessing, -- Cycle 3: processing row 0
              MProcessing, -- Cycle 4
              MProcessing, -- Cycle 5
              MProcessing, -- Cycle 6
              MProcessing, -- Cycle 7: still processing when rowDone arrives
              MReset, -- Cycle 8: reset row 1
              MProcessing, -- Cycle 9: processing row 1
              MProcessing, -- Cycle 10
              MProcessing, -- Cycle 11
              MProcessing, -- Cycle 12
              MProcessing, -- Cycle 13: still processing when rowDone arrives
              MReset, -- Cycle 14: reset row 2
              MProcessing, -- Cycle 15: processing row 2
              MProcessing, -- Cycle 16
              MProcessing, -- Cycle 17
              MProcessing, -- Cycle 18
              MProcessing -- Cycle 19: still processing when rowDone arrives (last row)
            ]

          expectedRowResets =
            [ False,
              False,
              True,
              False,
              False,
              False,
              False,
              False,
              True,
              False,
              False,
              False,
              False,
              False,
              True,
              False,
              False,
              False,
              False,
              False
            ]

          expectedRowEnables =
            [ False,
              False,
              False,
              True,
              True,
              True,
              True,
              False,
              False,
              True,
              True,
              True,
              True,
              False,
              False,
              True,
              True,
              True,
              True,
              False
            ]

          expectedValidOuts =
            P.replicate 20 False P.++ [True] -- validOut on cycle 20 (after last rowDone)
          expectedReadyOuts =
            [True, True] P.++ P.replicate 19 False P.++ [True] -- ready at start and after completion
      it "transitions through states correctly" $ do
        P.take 20 states `shouldBe` expectedStates

      it "generates rowReset at correct cycles" $ do
        rowResets `shouldBe` expectedRowResets

      it "generates rowEnable at correct cycles" $ do
        rowEnables `shouldBe` expectedRowEnables

      it "asserts validOut after last row completes" $ do
        P.take (P.length expectedValidOuts + 1) validOuts `shouldBe` expectedValidOuts

      it "asserts readyOut only when idle" $ do
        P.take (P.length expectedReadyOuts + 2) readyOuts `shouldBe` expectedReadyOuts

      it "completes full matrix multiplication cycle" $ do
        let doneIndices = DL.findIndices id validOuts
        P.length doneIndices `shouldBe` 1
        DL.head doneIndices `shouldBe` 20

    describe "matrixMultiplierStateMachine" $ do
        context "correctly handles state transitions and control signals (assuming 3 rows and 4 columns)" $ do
          let maxCycles = 25
              -- Simulate processing: each row takes (numCols + 1) cycles
              -- Cycle 0: idle
              -- Cycle 1: enable goes high, transition to MReset
              -- Cycle 2: MReset (rowReset=True)
              -- Cycle 3-6: MProcessing (rowEnable=True), row 0
              -- Cycle 7: rowDone for row 0, transition to MReset
              -- Cycle 8: MReset (rowReset=True)
              -- Cycle 9-12: MProcessing, row 1
              -- Cycle 13: rowDone for row 1, transition to MReset
              -- Cycle 14: MReset (rowReset=True)
              -- Cycle 15-18: MProcessing, row 2
              -- Cycle 19: rowDone for row 2, transition to MDone
              -- Cycle 20: MDone, waiting for downstream (readyIn=False)
              -- Cycle 21: MDone, waiting for downstream (readyIn=False)
              -- Cycle 22: MDone, downstream ready (readyIn=True), transition to MIdle
              -- Cycle 23: MIdle, ready for next input

              -- Input: enable signal (high on cycle 1)
              enableStream =
                [False, True] P.++ P.replicate (maxCycles - 2) False
              enable :: Signal System Bool
              enable = fromList enableStream

              -- Input: rowDone signal (pulses after each row completes)
              -- Row processing takes numCols cycles, done appears 1 cycle after last enable
              rowDoneStream =
                P.replicate 7 False
                  P.++ [True] -- Row 0 done at cycle 7
                  P.++ P.replicate 5 False
                  P.++ [True] -- Row 1 done at cycle 13
                  P.++ P.replicate 5 False
                  P.++ [True] -- Row 2 done at cycle 19
                  P.++ P.replicate (maxCycles - 20) False
              rowDone :: Signal System Bool
              rowDone = fromList rowDoneStream

              -- Input: currentRow counter (0, 1, 2 cycling through rows)
              currentRowStream =
                P.replicate 8 0 -- Row 0 until cycle 7
                  P.++ P.replicate 6 1 -- Row 1 until cycle 13
                  P.++ P.replicate 6 2 -- Row 2 until cycle 19
                  P.++ P.replicate (maxCycles - 20) 2 -- Stay at row 2
              currentRow :: Signal System (Index 3)
              currentRow = fromList currentRowStream

              -- Input: readyIn signal (downstream backpressure)
              -- Downstream not ready for 2 cycles after completion, then ready
              readyInStream =
                P.replicate 20 True -- Always ready during processing
                  P.++ [False, False, True] -- Backpressure for 2 cycles, then ready
                  P.++ P.replicate (maxCycles - 23) True
              readyIn :: Signal System Bool
              readyIn = fromList readyInStream

              (state, rowReset, rowEnable, validOut, readyOut) =
                exposeClockResetEnable
                  matrixMultiplierStateMachine
                  CS.systemClockGen
                  CS.resetGen
                  CS.enableGen
                  enable
                  readyIn
                  rowDone
                  currentRow

              states = P.take maxCycles $ sample state
              rowResets = P.take maxCycles $ sample rowReset
              rowEnables = P.take maxCycles $ sample rowEnable
              validOuts = P.take (maxCycles + 1) $ sample validOut
              readyOuts = P.take (maxCycles + 2) $ sample readyOut

              expectedStates =
                [ MIdle, -- Cycle 0: initial idle
                  MIdle, -- Cycle 1: enable received, but state updates next cycle
                  MReset, -- Cycle 2: reset row 0
                  MProcessing, -- Cycle 3: processing row 0
                  MProcessing, -- Cycle 4
                  MProcessing, -- Cycle 5
                  MProcessing, -- Cycle 6
                  MProcessing, -- Cycle 7: still processing when rowDone arrives
                  MReset, -- Cycle 8: reset row 1
                  MProcessing, -- Cycle 9: processing row 1
                  MProcessing, -- Cycle 10
                  MProcessing, -- Cycle 11
                  MProcessing, -- Cycle 12
                  MProcessing, -- Cycle 13: still processing when rowDone arrives
                  MReset, -- Cycle 14: reset row 2
                  MProcessing, -- Cycle 15: processing row 2
                  MProcessing, -- Cycle 16
                  MProcessing, -- Cycle 17
                  MProcessing, -- Cycle 18
                  MProcessing, -- Cycle 19: still processing when rowDone arrives (last row)
                  MDone, -- Cycle 20: done, waiting for downstream
                  MDone, -- Cycle 21: still waiting (readyIn=False)
                  MDone, -- Cycle 22: still waiting (readyIn=False)
                  MIdle, -- Cycle 23: downstream accepted, back to idle
                  MIdle -- Cycle 24: idle
                ]

              expectedRowResets =
                [ False,
                  False,
                  True, -- Cycle 2
                  False,
                  False,
                  False,
                  False,
                  False,
                  True, -- Cycle 8
                  False,
                  False,
                  False,
                  False,
                  False,
                  True, -- Cycle 14
                  False,
                  False,
                  False,
                  False,
                  False,
                  False,
                  False,
                  False,
                  False,
                  False
                ]

              expectedRowEnables =
                [ False,
                  False,
                  False,
                  True, -- Cycle 3-6: row 0 processing
                  True,
                  True,
                  True,
                  False, -- Cycle 7: rowDone high, disable
                  False,
                  True, -- Cycle 9-12: row 1 processing
                  True,
                  True,
                  True,
                  False, -- Cycle 13: rowDone high, disable
                  False,
                  True, -- Cycle 15-18: row 2 processing
                  True,
                  True,
                  True,
                  False, -- Cycle 19: rowDone high, disable
                  False,
                  False,
                  False,
                  False,
                  False
                ]

              expectedValidOuts =
                P.replicate 20 False
                  P.++ [True, True, True] -- validOut asserted at cycle 20-22 (MDone)
                  P.++ P.replicate 2 False -- validOut drops after returning to idle

              expectedReadyOuts =
                [True, True] -- Cycle 0-1: ready (idle)
                  P.++ P.replicate 21 False -- Cycle 2-22: not ready (processing/done)
                  P.++ [True, True] -- Cycle 23-24: ready again (idle)

          it "transitions through states correctly" $ do
            states `shouldBe` expectedStates

          it "generates rowReset at correct cycles" $ do
            rowResets `shouldBe` expectedRowResets

          it "generates rowEnable at correct cycles" $ do
            rowEnables `shouldBe` expectedRowEnables

          it "asserts validOut when done and maintains until consumed" $ do
            P.take (P.length expectedValidOuts) validOuts `shouldBe` expectedValidOuts

          it "asserts readyOut only when idle" $ do
            P.take (P.length expectedReadyOuts) readyOuts `shouldBe` expectedReadyOuts

          it "handles downstream backpressure correctly" $ do
            -- Should stay in MDone state while readyIn is False
            states P.!! 20 `shouldBe` MDone
            states P.!! 21 `shouldBe` MDone
            states P.!! 22 `shouldBe` MDone
            -- Should transition to MIdle when readyIn becomes True
            states P.!! 23 `shouldBe` MIdle

          it "maintains validOut during backpressure" $ do
            -- validOut should stay high while in MDone
            validOuts P.!! 20 `shouldBe` True
            validOuts P.!! 21 `shouldBe` True
            validOuts P.!! 22 `shouldBe` True
            -- validOut should drop after transitioning to MIdle
            validOuts P.!! 23 `shouldBe` False

          it "completes full matrix multiplication cycle with backpressure" $ do
            let doneIndices = DL.elemIndices MDone states
            P.length doneIndices `shouldBe` 3 -- Cycles 20, 21, 22
            DL.head doneIndices `shouldBe` 20

  describe "matrixMultiplier" $ do
    context "computes matrix-vector multiplication using handshake protocol" $ do
      let maxCycles = 30

          -- Define a 3x4 matrix
          -- Row 0: [2, 1, 3, 2]
          -- Row 1: [1, 2, 1, 3]
          -- Row 2: [3, 2, 2, 1]
          row0 :: RowI8E 4
          row0 = (2 :> 1 :> 3 :> 2 :> Nil, 0)

          row1 :: RowI8E 4
          row1 = (1 :> 2 :> 1 :> 3 :> Nil, 0)

          row2 :: RowI8E 4
          row2 = (3 :> 2 :> 2 :> 1 :> Nil, 0)

          matrix :: MatI8E 3 4
          matrix = row0 :> row1 :> row2 :> Nil

          -- Input vector: [2.0, 1.5, 1.0, 0.5]
          inputVector :: Vec 4 FixedPoint
          inputVector = 2.0 :> 1.5 :> 1.0 :> 0.5 :> Nil

          -- Expected outputs:
          -- Row 0: 2*2.0 + 1*1.5 + 3*1.0 + 2*0.5 = 4.0 + 1.5 + 3.0 + 1.0 = 9.5
          -- Row 1: 1*2.0 + 2*1.5 + 1*1.0 + 3*0.5 = 2.0 + 3.0 + 1.0 + 1.5 = 7.5
          -- Row 2: 3*2.0 + 2*1.5 + 2*1.0 + 1*0.5 = 6.0 + 3.0 + 2.0 + 0.5 = 11.5
          expectedResult :: Vec 3 FixedPoint
          expectedResult = 9.5 :> 7.5 :> 11.5 :> Nil

          tolerance = 0.01

          -- Input signals
          -- validIn: pulse high only on cycle 1 (when ready to start transaction)
          validIn :: Signal System Bool
          validIn = fromList $ [False, True] P.++ P.replicate (maxCycles - 2) False

          readyIn :: Signal System Bool
          readyIn = pure True  -- previous output always considered as consumed

          (outputVec, validOut, readyOut) =
            exposeClockResetEnable
              matrixMultiplier
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              validIn
              readyIn
              matrix
              (pure inputVector)

          outputs = P.take maxCycles $ sample outputVec
          validOuts = P.take maxCycles $ sample validOut
          readyOuts = P.take maxCycles $ sample readyOut

          -- Find when validOut goes high
          validIndices = DL.findIndices id validOuts
          completionCycle = if null validIndices then 0 else DL.head validIndices
          finalOutput = if null validIndices then repeat 0 else outputs P.!! completionCycle

      it "asserts readyOut initially (ready to accept transaction)" $ do
        DL.head readyOuts `shouldBe` True

      it "only one valid pulse emitted" $ do
        P.length validIndices `shouldBe` 1

      it "produces correct result vector when validOut is asserted" $ do
        let matches =
              P.zipWith
                (\a e -> abs (a - e) < tolerance)
                (toList finalOutput)
                (toList expectedResult)
        DL.and matches `shouldBe` True

      it "returns to ready state after completion" $ do
        if completionCycle < maxCycles - 1
          then readyOuts P.!! (completionCycle + 1) `shouldBe` True
          else True `shouldBe` False -- can't test if at end of sampled cycles

