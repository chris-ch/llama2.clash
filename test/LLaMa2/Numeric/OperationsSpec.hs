module LLaMa2.Numeric.OperationsSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Numeric.Operations (accumulator, MultiplierState (..),
  matrixMultiplierStateMachine, cyclicalCounter64, parallel64RowProcessor)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Numeric.Types (FixedPoint)
import Test.Hspec
import qualified Prelude as P

-- ==========
-- Simulation Helpers
-- ==========

spec :: Spec
spec = do
  describe "Accumulator" $ do
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

  describe "Cyclical Counter 64" $ do
    context "correctly increments, resets, and holds the counter" $ do
      let maxCycles = 11
          resetStream = [True, False, False, False, True, False, False, True, False, False, False]
          enableStream = [False, True, True, True, False, True, False, True, True, True, False]
          expected = [0, 0, 64, 128, 192, 0, 64, 64, 0, 64, 128] -- Increment by 64 when enabled
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              cyclicalCounter64
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
          outputs = P.take maxCycles $ sample counter :: [Index 256]
          outputInts = P.map fromEnum outputs
      it "output stream as expected" $ do
        outputInts `shouldBe` expected
      it "does not exceed max value" $ do
        let maxBoundVal = fromEnum (maxBound :: Index 256)
        all (<= maxBoundVal) outputInts `shouldBe` True

    context "ignores enable when reset is active" $ do
      let maxCycles = 11
          resetStream = [True, False, True, False, True, False, True, False, True, False, False]
          enableStream = [True, True, True, True, True, True, True, True, True, True, True]
          expected = [0, 0, 64, 0, 64, 0, 64, 0, 64, 0, 64] -- Reset overrides enable
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              cyclicalCounter64
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
          outputs = P.take maxCycles $ sample counter :: [Index 256]
          outputInts = P.map fromEnum outputs
      it "output stream as expected" $ do
        outputInts `shouldBe` expected
      it "does not exceed max value" $ do
        let maxBoundVal = fromEnum (maxBound :: Index 256)
        all (<= maxBoundVal) outputInts `shouldBe` True

  describe "Row Processor" $ do
    context "computes dot product for a subset of rows (parallel 64 version)" $ do
      let maxCycles = 12

          rowVector :: RowI8E 4
          rowVector = RowI8E {rowMantissas = 1 :> 2 :> 3 :> 4 :> Nil, rowExponent = 0}

          columnVector :: Vec 4 FixedPoint
          columnVector = 1.0 :> 0.5 :> 0.25 :> 0.125 :> Nil

          -- Input signals
          row :: Signal System (RowI8E 4)
          row = pure rowVector

          resets = True : P.replicate (maxCycles - 1) False
          reset :: Signal System Bool
          reset = fromList resets

          enables = [False] P.++ P.replicate 4 True P.++ P.replicate (maxCycles - 5) False
          enable :: Signal dom Bool
          enable = fromList enables

          -- make sure to pad first cycle for reset warm-up
          column :: Signal dom (Vec 4 FixedPoint)
          column =
            fromList $ DL.replicate 5 columnVector
              P.++ DL.repeat (pure 0)

          (outputComponent, rowDone, _) =
            exposeClockResetEnable
              parallel64RowProcessor
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

      it "there should be only one completion" $ do
        P.length doneIndices `shouldBe` 4
      it "(test input) reset flags should match expected flags" $ do
        let expectedResets = [True,False,False,False,False,False,False,False,False,False,False,False]
        resets `shouldBe` expectedResets
      it "(test input) enable flags should match expected flags" $ do
        let expectedEnables = [False,True,True,True,True,False,False,False,False,False,False,False]
        enables `shouldBe` expectedEnables
      it "done flags should match expected flags" $ do
        let expectedDones = [False,False,True,True,True,True,False,False,False,False,False,False]
        dones `shouldBe` expectedDones
      it "final output should match expected value" $ do
        let expectedOuts = [0.0,0.0,3.25,3.25,3.25,3.25,3.25,3.25,3.25,3.25,3.25,3.25]
        outs `shouldBe` expectedOuts

    context "computes dot product for a 3-column row" $ do
      let maxCycles = 7
          rowVector :: RowI8E 3
          rowVector = RowI8E {rowMantissas = 1 :> 2 :> 3 :> Nil, rowExponent = 0}
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
          (outputComponent, rowDone , _) =
            exposeClockResetEnable
              parallel64RowProcessor
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
      it "completes at expected cycle" $ do
        doneIndices `shouldBe` [3,4,5]
      it "produces correct dot product" $ do
        abs (finalOut - expected) < tolerance `shouldBe` True
      it "accumulator follows expected sequence" $ do
        let
          expectedOuts = [0.0,0.0,0.0,14.0,14.0,14.0,14.0]
          actualOutputs = P.take maxCycles outs
        P.all (\(a, e) -> abs (a - e) < tolerance) (P.zip actualOutputs expectedOuts) `shouldBe` True

    context "computes dot product for a 3-column row, independently from components before enable" $ do
      let maxCycles = 7
          rowVector :: RowI8E 3
          rowVector = RowI8E {rowMantissas = 1 :> 2 :> 3 :> Nil, rowExponent = 0}
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

          (outputComponent, rowDone ,_) =
            exposeClockResetEnable
              parallel64RowProcessor
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
        doneIndices `shouldBe` [3,4,5]
      it "produces correct dot product" $ do
        abs (finalOut - expected) < tolerance `shouldBe` True
      it "accumulator follows expected sequence" $ do
        let expectedOuts = [0.0,0.0,0.0,14.0,14.0,14.0,14.0]
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
          validIn :: Signal System Bool
          validIn = fromList enableStream

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

          (state, _, rowReset, rowEnable, validOut, readyOut) =
            exposeClockResetEnable
              matrixMultiplierStateMachine
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              validIn
              (pure True)
              readyIn
              rowDone
              currentRow

          states = P.take maxCycles $ sample state
          rowResets = P.take maxCycles $ sample rowReset
          rowEnables = P.take maxCycles $ sample rowEnable
          validOuts = P.take (maxCycles + 1) $ sample validOut
          readyOuts = P.take (maxCycles + 2) $ sample readyOut

          expectedStates =
            [MIdle,MIdle,MFetching,MReset,MProcessing,MProcessing,MProcessing,MProcessing,MFetching,MReset,MProcessing,MProcessing,MProcessing,MProcessing,MFetching,MReset,MProcessing,MProcessing,MProcessing,MProcessing]

          expectedRowResets =
            [False,False,False,True,False,False,False,False,False,True,False,False,False,False,False,True,False,False,False,False]

          expectedRowEnables =
            [False,False,False,False,True,True,True,False,False,False,True,True,True,False,False,False,True,True,True,False]

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
              validIn :: Signal System Bool
              validIn = fromList enableStream

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

              (state, _, rowReset, rowEnable, validOut, readyOut) =
                exposeClockResetEnable
                  matrixMultiplierStateMachine
                  CS.systemClockGen
                  CS.resetGen
                  CS.enableGen
                  validIn
                  (pure True)
                  readyIn
                  rowDone
                  currentRow

              states = P.take maxCycles $ sample state
              rowResets = P.take maxCycles $ sample rowReset
              rowEnables = P.take maxCycles $ sample rowEnable
              validOuts = P.take (maxCycles + 1) $ sample validOut
              readyOuts = P.take (maxCycles + 2) $ sample readyOut

              expectedStates =
                [MIdle,MIdle,MFetching,MReset,MProcessing,MProcessing,MProcessing,MProcessing,MFetching,MReset,MProcessing,MProcessing,MProcessing,MProcessing,MFetching,MReset,MProcessing,MProcessing,MProcessing,MProcessing,MDone,MDone,MDone,MIdle,MIdle]

              expectedRowResets =
                [False,False,False,True,False,False,False,False,False,True,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False]

              expectedRowEnables =
                [False,False,False,False,True,True,True,False,False,False,True,True,True,False,False,False,True,True,True,False,False,False,False,False,False]

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
    context "handles reset overriding enable (size = 64)" $ do
      let maxCycles = 10
          resetStream = [True, False, False, True, False, False, True, False, False, False]
          enableStream = [True, True, True, True, True, True, True, True, True, True]
          expected = [0,0,63,63,0,63,63,0,63,63] :: [Int]
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              cyclicalCounter64
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
          outputs = P.take maxCycles $ sample counter :: [Index 64]
          outputInts = P.map fromEnum outputs
          maxBoundVal = fromEnum (maxBound :: Index 64)
      it "output stream matches expected sequence" $ do
        outputInts `shouldBe` expected
      it "never exceeds maxBound (63)" $ do
        all (<= maxBoundVal) outputInts `shouldBe` True

    context "holds value when enable is low (size = 64)" $ do
      let maxCycles = 10
          resetStream = [True, False, False, False, False, True, False, False, False, False]
          enableStream = [False, True, False, True, False, False, True, False, True, False]
          expected = [0,0,63,63,63,63,0,63,63,63] :: [Int]
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              cyclicalCounter64
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
          outputs = P.take maxCycles $ sample counter :: [Index 64]
          outputInts = P.map fromEnum outputs
          maxBoundVal = fromEnum (maxBound :: Index 64)
      it "output stream matches expected sequence" $ do
        outputInts `shouldBe` expected
      it "never exceeds maxBound (63)" $ do
        all (<= maxBoundVal) outputInts `shouldBe` True

    context "increments correctly for larger size (size = 128)" $ do
      let maxCycles = 10
          resetStream = [True, False, False, False, False, True, False, False, False, False]
          enableStream = [False, True, True, True, True, False, True, True, True, True]
          expected = [0,0,64,127,127,127,0,64,127,127] :: [Int]
          reset = fromList resetStream :: Signal System Bool
          enable = fromList enableStream :: Signal System Bool
          counter =
            exposeClockResetEnable
              cyclicalCounter64
              CS.systemClockGen
              CS.resetGen
              CS.enableGen
              reset
              enable
          outputs = P.take maxCycles $ sample counter :: [Index 128]
          outputInts = P.map fromEnum outputs
          maxBoundVal = fromEnum (maxBound :: Index 128)
      it "output stream matches expected sequence" $ do
        outputInts `shouldBe` expected
      it "never exceeds maxBound (127)" $ do
        all (<= maxBoundVal) outputInts `shouldBe` True
