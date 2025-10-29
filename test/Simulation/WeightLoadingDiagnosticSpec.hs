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
