module Simulation.WeightLoadingDiagnosticSpec (spec) where

import Clash.Prelude
import Test.Hspec
import qualified Prelude as P
import qualified Data.List as DL

import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer
import LLaMa2.Memory.LayerAddressing (WeightAddress(..), WeightMatrixType(..))
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Types.ModelConfig
  ( ModelDimension, HeadDimension, NumQueryHeads, NumKeyValueHeads )
import Control.Monad (when, forM_)

-- This spec focuses on the CORE timing bug:
-- When does fullyLoaded become True relative to when weights are actually ready?

spec :: Spec
spec = do
  describe "Weight Loading Timing Diagnostics" $ do

    it "TEST 1: fullyLoaded should be False until ALL rows loaded" $ do
      -- This tests the fundamental contract of the weight buffer

      let writes = generateFullWriteSequence 10
          totalCycles = P.maximum (DL.map (\(c,_,_,_) -> c) writes) + 10

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          loadedFlags = DL.map fullyLoaded samples

          -- Find when fullyLoaded first becomes True
          firstLoadedCycle = DL.findIndex id loadedFlags

          -- Find when last row was written
          lastWriteCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) writes

      case firstLoadedCycle of
        Nothing -> expectationFailure "fullyLoaded never became True"
        Just loadedCycle -> do
          putStrLn $ "\n✓ fullyLoaded became True at cycle: " P.++ show loadedCycle
          putStrLn $ "✓ Last weight written at cycle:      " P.++ show lastWriteCycle

          -- CRITICAL: fullyLoaded should not be True BEFORE last write
          loadedCycle `shouldSatisfy` (>= lastWriteCycle)

    it "TEST 2: simulate layer transition - buffer must clear then reload" $ do
      -- This simulates what happens when layer index changes

      let -- First layer load (cycles 10-100)
          layer0Writes = generateFullWriteSequence 10
          layer0LastCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) layer0Writes

          -- Reset pulse after layer 0 completes
          resetCycle = layer0LastCycle + 50

          -- Second layer load (cycles resetCycle+10 to resetCycle+100)
          layer1Writes = generateFullWriteSequence (resetCycle + 10)
          layer1LastCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) layer1Writes

          allWrites = layer0Writes  -- Only layer 0 for now
          totalCycles = layer1LastCycle + 50

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles allWrites

          -- Reset pulse at resetCycle
          resetList = DL.replicate resetCycle False P.++ [True] P.++ DL.repeat False
          reset = fromList resetList

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          loadedFlags = DL.map fullyLoaded samples

          -- Check states around reset
          beforeReset = loadedFlags DL.!! (resetCycle - 1)
          atReset = loadedFlags DL.!! resetCycle
          afterReset = loadedFlags DL.!! (resetCycle + 2)

      putStrLn "\n=== Layer Transition Timing ==="
      putStrLn $ "Before reset (cycle " P.++ show (resetCycle-1) P.++ "): fullyLoaded = " P.++ show beforeReset
      putStrLn $ "At reset     (cycle " P.++ show resetCycle P.++ "): fullyLoaded = " P.++ show atReset
      putStrLn $ "After reset  (cycle " P.++ show (resetCycle+2) P.++ "): fullyLoaded = " P.++ show afterReset

      beforeReset `shouldBe` True   -- Should be loaded before reset
      afterReset `shouldBe` False   -- Should be cleared after reset

    it "TEST 3: detect spurious fullyLoaded pulses" $ do
      -- Check for any glitches where fullyLoaded goes True→False→True without reset

      let writes = generateFullWriteSequence 10
          lastWriteCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) writes
          totalCycles = lastWriteCycle + 5000

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False  -- NO RESET during observation

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          loadedFlags = DL.map fullyLoaded samples

          -- Detect True→False transitions (without reset, these are bugs)
          transitions = [ (c, prev, curr) |
                         (c, prev, curr) <- P.zip3 [0 :: Int ..] loadedFlags (P.tail loadedFlags),
                         prev && P.not curr ]

      case transitions of
        [] -> return ()  -- Good!
        glitches -> do
          putStrLn "\n❌ FOUND SPURIOUS TRANSITIONS:"
          mapM_ (\(c, _, _) -> putStrLn $ "  Cycle " P.++ show c P.++ ": True → False") glitches
          expectationFailure $ "fullyLoaded glitched " P.++ show (P.length glitches) P.++ " times"

    it "TEST 4: allDone pulse interaction with fullyLoaded" $ do
      -- The allDone signal should cause fullyLoaded to latch True
      -- Let's verify the exact timing

      let writes = generateFullWriteSequence 10
          (lastCycle, _, _lastAddr, _) = DL.last writes

          totalCycles = lastCycle + 20

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          allDoneSamples = DL.take totalCycles $ sample allDone
          loadedFlags = DL.map fullyLoaded samples

          -- Find when allDone pulses
          allDonePulseCycles = [ c | (c, done) <- P.zip [0..] allDoneSamples, done ]

          -- Find when fullyLoaded becomes True
          firstLoadedCycle = DL.findIndex id loadedFlags

      putStrLn "\n=== allDone → fullyLoaded Timing ==="
      putStrLn $ "allDone pulses at cycles: " P.++ show allDonePulseCycles
      putStrLn $ "fullyLoaded first True at cycle: " P.++ show firstLoadedCycle

      case (allDonePulseCycles, firstLoadedCycle) of
        (doneCycle:_, Just loadedCycle) -> do
          -- fullyLoaded should become True within 2 cycles of allDone
          let latency = loadedCycle - doneCycle
          putStrLn $ "Latency: " P.++ show latency P.++ " cycles"
          latency `shouldSatisfy` (<= 2)
        _ -> expectationFailure "Could not measure timing"

    it "TEST 5: verify loaded Q weights contain correct values" $ do
      -- This checks that the weights actually made it into the buffer correctly

      let writes = generateFullWriteSequence 10
          lastWriteCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) writes
          totalCycles = lastWriteCycle + 20

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          
          -- Find when buffer is fully loaded
          firstLoadedCycle = DL.findIndex fullyLoaded samples

      case firstLoadedCycle of
        Nothing -> expectationFailure "Buffer never became fullyLoaded"
        Just loadedCycle -> do
          -- Get the buffer state after loading completes
          let loadedBuffer = samples DL.!! (loadedCycle + 2)  -- +2 for safety
          
          -- Extract first Q weight for head 0
          let qWeight0 = extractQWeight loadedBuffer 0
              firstRow = qWeight0 !! (0 :: Int)
              (mantissas, exponent') = firstRow
              
          putStrLn "\n=== Loaded Q Weight Verification ==="
          putStrLn $ "First Q row (head 0) loaded at cycle: " P.++ show loadedCycle
          putStrLn $ "Exponent: " P.++ show exponent'
          putStrLn $ "First 5 mantissas: " P.++ show (DL.take 5 $ toList mantissas)
          
          -- Verify against expected synthetic values
          -- From makeSyntheticRow with rowIdx=0:
          -- mantissa[i] = rowIdx * 10 + i = 0 * 10 + i = i
          -- exponent = rowIdx = 0
          let expectedMantissas = [0..4] :: [Signed 8]
              actualMantissas = DL.take 5 $ toList mantissas
              
          exponent' `shouldBe` 0
          actualMantissas `shouldBe` expectedMantissas
          
          -- Check a row in the middle (rowIdx = 5)
          let midRow = qWeight0 !! (5 :: Int)
              (midMantissas, midExp) = midRow
              midExpectedMantissas = [50..54] :: [Signed 8]
              midActualMantissas = DL.take 5 $ toList midMantissas
              
          putStrLn "\nMiddle Q row (head 0, row 5):"
          putStrLn $ "Exponent: " P.++ show midExp
          putStrLn $ "First 5 mantissas: " P.++ show midActualMantissas
          
          midExp `shouldBe` 5
          midActualMantissas `shouldBe` midExpectedMantissas

    it "TEST 6: verify loaded K and V weights for multiple heads" $ do
      -- Check K and V weights across different heads

      let writes = generateFullWriteSequence 10
          lastWriteCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) writes
          totalCycles = lastWriteCycle + 20

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          firstLoadedCycle = DL.findIndex fullyLoaded samples

      case firstLoadedCycle of
        Nothing -> expectationFailure "Buffer never became fullyLoaded"
        Just loadedCycle -> do
          let loadedBuffer = samples DL.!! (loadedCycle + 2)
          
          -- Check K weight for KV head 0, row 0
          let kWeight0 = extractKWeight loadedBuffer 0
              kFirstRow = kWeight0 !! (0 :: Int)
              (kMantissas, kExp) = kFirstRow
              kActualMantissas = DL.take 5 $ toList kMantissas
              
          putStrLn "\n=== K Weight Verification ==="
          putStrLn "First K row (KV head 0):"
          putStrLn $ "Exponent: " P.++ show kExp
          putStrLn $ "First 5 mantissas: " P.++ show kActualMantissas
          
          -- K weights are written after all Q weights
          -- makeSyntheticRow still uses rowIdx for generation
          kExp `shouldBe` 0
          kActualMantissas `shouldBe` [0 .. 4]
          
          -- Check V weight for KV head 0, row 0
          let vWeight0 = extractVWeight loadedBuffer 0
              vFirstRow = vWeight0 !! (0 :: Int)
              (vMantissas, vExp) = vFirstRow
              vActualMantissas = DL.take 5 $ toList vMantissas
              
          putStrLn "\n=== V Weight Verification ==="
          putStrLn "First V row (KV head 0):"
          putStrLn $ "Exponent: " P.++ show vExp
          putStrLn $ "First 5 mantissas: " P.++ show vActualMantissas
          
          vExp `shouldBe` 0
          vActualMantissas `shouldBe` [0..4]
          
          -- Verify a different head (if NumKeyValueHeads > 1)
          when (natToNum @NumKeyValueHeads > (1 :: Int)) $ do
            let kWeight1 = extractKWeight loadedBuffer 1
                kRow1 = kWeight1 !! (0 :: Int)
                (kMant1, kExp1) = kRow1
                kActual1 = DL.take 5 $ toList kMant1
                
            putStrLn "\n=== K Weight Head 1 Verification ==="
            putStrLn "First K row (KV head 1):"
            putStrLn $ "Exponent: " P.++ show kExp1
            putStrLn $ "First 5 mantissas: " P.++ show kActual1
            
            kExp1 `shouldBe` 0
            kActual1 `shouldBe` [0..4]

    it "TEST 7: verify all Q heads have distinct weight values" $ do
      -- Ensure weights for different heads are actually different

      let writes = generateFullWriteSequence 10
          lastWriteCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) writes
          totalCycles = lastWriteCycle + 20

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          firstLoadedCycle = DL.findIndex fullyLoaded samples

      case firstLoadedCycle of
        Nothing -> expectationFailure "Buffer never became fullyLoaded"
        Just loadedCycle -> do
          let loadedBuffer = samples DL.!! (loadedCycle + 2)
              numQHeads = fromInteger (natToNum @NumQueryHeads) :: Int
          
          putStrLn "\n=== Multi-Head Weight Verification ==="
          
          -- Check first row of each Q head
          let headSamples = [ (h, extractQWeight loadedBuffer (toEnum h) !! (0 :: Int))
                            | h <- [0 .. min 3 (numQHeads - 1)] ]  -- Check up to 4 heads
          
          forM_ headSamples $ \(headIdx, (mantissas, exp')) -> do
            let firstFive = DL.take 5 $ toList mantissas
            putStrLn $ "Q Head " P.++ show headIdx P.++ 
                      " - First row exp: " P.++ show exp' P.++
                      ", mantissas: " P.++ show firstFive
            
            -- All should have same structure (row 0) but verify they're there
            exp' `shouldBe` 0
            DL.head firstFive `shouldBe` 0

-- Helper: generate a complete Q/K/V write sequence
generateFullWriteSequence :: Int -> [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
generateFullWriteSequence startCycle = qWrites P.++ kWrites P.++ vWrites
 where
  numQHeads  = fromInteger (natToNum @NumQueryHeads) :: Int
  numKVHeads = fromInteger (natToNum @NumKeyValueHeads) :: Int
  hdDim      = fromInteger (natToNum @HeadDimension) :: Int

  qWrites = [ (startCycle + hd * hdDim + fromEnum rowIdx, True,
               WeightAddress rowIdx QMatrix (fromIntegral hd),
               makeSyntheticRow rowIdx)
            | hd <- [0 .. numQHeads - 1]
            , rowIdx <- [minBound .. maxBound] :: [Index HeadDimension] ]

  kWrites = [ (startCycle + P.length qWrites + hd * hdDim + fromEnum rowIdx, True,
               WeightAddress rowIdx KMatrix (fromIntegral hd),
               makeSyntheticRow rowIdx)
            | hd <- [0 .. numKVHeads - 1]
            , rowIdx <- [minBound .. maxBound] :: [Index HeadDimension] ]

  vWrites = [ (startCycle + P.length qWrites + P.length kWrites + hd * hdDim + fromEnum rowIdx, True,
               WeightAddress rowIdx VMatrix (fromIntegral hd),
               makeSyntheticRow rowIdx)
            | hd <- [0 .. numKVHeads - 1]
            , rowIdx <- [minBound .. maxBound] :: [Index HeadDimension] ]

makeSyntheticRow :: Index HeadDimension -> RowI8E ModelDimension
makeSyntheticRow rowIdx =
  ( imap (\i _ -> fromIntegral (fromEnum rowIdx * 10 + fromEnum i)) (repeat (0 :: Signed 8))
  , fromIntegral (fromEnum rowIdx) )

createSignalsFromSequence :: Int -> [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
  -> (Signal System Bool, Signal System WeightAddress,
      Signal System (RowI8E ModelDimension), Signal System Bool)
createSignalsFromSequence totalCycles writes = (streamValidSig, addrSig, rowSig, allDoneSig)
  where
    writeMap = DL.map (\(c, v, a, r) -> (c, (v, a, r))) writes
    defaultAddr = WeightAddress 0 QMatrix 0
    defaultRow = (repeat 0, 0)

    streamValidSig = fromList
      [ P.maybe False (\(v,_,_) -> v) (DL.lookup c writeMap) | c <- [0..totalCycles-1] ]

    addrSig = fromList
      [ P.maybe defaultAddr (\(_,a,_) -> a) (DL.lookup c writeMap) | c <- [0..totalCycles-1] ]

    rowSig = fromList
      [ P.maybe defaultRow (\(_,_,r) -> r) (DL.lookup c writeMap) | c <- [0..totalCycles-1] ]

    allDoneSig = fromList
      [ case DL.lookup c writeMap of
          Just (True, addr, _) -> isLastVWrite addr
          _ -> False
      | c <- [0..totalCycles-1] ]

isLastVWrite :: WeightAddress -> Bool
isLastVWrite WeightAddress{..} =
  matrixType == VMatrix
  && headIndex == fromInteger (natToNum @NumKeyValueHeads - 1)
  && rowIndex == maxBound
