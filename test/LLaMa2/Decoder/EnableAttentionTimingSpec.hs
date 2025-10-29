module LLaMa2.Decoder.EnableAttentionTimingSpec (spec) where

import Clash.Prelude
import Test.Hspec
import qualified Prelude as P
import qualified Data.List as DL

import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer
import LLaMa2.Memory.LayerAddressing (WeightAddress(..), WeightMatrixType(..))
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Types.ModelConfig
  ( ModelDimension, HeadDimension, NumQueryHeads, NumKeyValueHeads )

import Text.Printf (printf)

-- This test simulates the EXACT scenario in Decoder.hs to find the timing bug

spec :: Spec
spec = do
  describe "Decoder enableAttention Timing" $ do

    it "CRITICAL: enableAttention must stay False during layer transition" $ do
      -- This simulates the exact logic in Decoder.hs:
      --   loadTrigger = firstCycle .||. layerChanged
      --   enableAttention = mux loadTrigger (pure False) (fullyLoaded <$> weightBuffer)

      let -- Simulate layer 0 loading and completing
          layer0Writes = generateFullWriteSequence 10
          layer0DoneCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) layer0Writes

          -- Simulate staying in layer 0 for a while
          layer0IdleCycles = 100

          -- At this cycle, layer changes from 0 to 1
          layerChangeCycle = layer0DoneCycle + layer0IdleCycles

          -- New layer 1 weights start loading
          layer1Writes = generateFullWriteSequence (layerChangeCycle + 10)
          layer1DoneCycle = P.maximum $ DL.map (\(c,_,_,_) -> c) layer1Writes

          allWrites = layer0Writes P.++ layer1Writes
          totalCycles = layer1DoneCycle + 100

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles allWrites

          -- Simulate firstCycle and layerChanged
          firstCycleSig = fromList $ True : DL.repeat False
          layerChangedSig = fromList $
            DL.replicate layerChangeCycle False P.++ [True] P.++ DL.repeat False

          -- This is loadTrigger from Decoder.hs
          loadTrigger = firstCycleSig .||. layerChangedSig

          -- Simulate reset based on loadTrigger (like in decoder)
          reset = loadTrigger

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          -- This is enableAttention from Decoder.hs
          enableAttention = mux loadTrigger (pure False) (fullyLoaded <$> bufferSig)

          -- Sample signals
          enableSamples = DL.take totalCycles $ sample enableAttention
          loadTriggerSamples = DL.take totalCycles $ sample loadTrigger
          fullyLoadedSamples = DL.take totalCycles $ sample (fullyLoaded <$> bufferSig)

          -- THE CRITICAL CHECK:
          -- After layerChangeCycle, enableAttention should stay False until
          -- layer 1 weights are fully loaded (layer1DoneCycle + 1)

          criticalWindow = [layerChangeCycle .. layer1DoneCycle]
          enableInCriticalWindow = [ (c, enable) |
                                    (c, enable) <- P.zip criticalWindow (P.drop layerChangeCycle enableSamples),
                                    enable ]

      putStrLn "\n=== Layer Transition Analysis ==="
      putStrLn $ "Layer change at cycle:      " P.++ show layerChangeCycle
      putStrLn $ "Layer 1 done at cycle:      " P.++ show layer1DoneCycle
      putStrLn $ "Critical window:            " P.++ show layerChangeCycle P.++ " to " P.++ show layer1DoneCycle

      -- Show detailed timeline around layer change
      putStrLn "\n=== Detailed Timeline ==="
      putStrLn "Cycle | loadTrig | fullyLoad | enableAtt"
      putStrLn "------+----------+-----------+----------"
      mapM_ (\c -> do
        let lt = loadTriggerSamples P.!! c
            fl = fullyLoadedSamples P.!! c
            ea = enableSamples P.!! c
        putStrLn $ printf "%5d | %8s | %9s | %9s"
          c (show lt) (show fl) (show ea)
        ) [layerChangeCycle - 5 .. min (totalCycles - 1) (layerChangeCycle + 50)]

      -- Check for violations
      case enableInCriticalWindow of
        [] -> putStrLn "\n✓ PASS: enableAttention stayed False during weight loading"
        violations -> do
          putStrLn "\n❌ FAIL: enableAttention was True during critical window!"
          putStrLn $ "Violations at cycles: " P.++ show (P.map fst violations)

          -- This is the BUG!
          expectationFailure $
            "enableAttention asserted while weights were loading. " P.++
            "This causes projection to run with stale/incorrect weights!"

    it "ROOT CAUSE: fullyLoaded from previous layer persists across reset" $ do
      -- This test checks if the buffer's fullyLoaded flag is causing the issue

      let writes1 = generateFullWriteSequence 10
          done1Cycle = P.maximum $ DL.map (\(c,_,_,_) -> c) writes1

          resetCycle = done1Cycle + 50

          allWrites = writes1  -- Only first load
          totalCycles = resetCycle + 100

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles allWrites

          -- Reset pulse
          resetSig = fromList $ DL.replicate resetCycle False P.++ [True] P.++ DL.repeat False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone resetSig

          fullyLoadedSamples = DL.take totalCycles $ sample (fullyLoaded <$> bufferSig)

          -- Check critical cycles
          beforeReset = fullyLoadedSamples P.!! (resetCycle - 1)
          atReset = fullyLoadedSamples P.!! resetCycle
          afterReset = fullyLoadedSamples P.!! (resetCycle + 1)
          wayAfterReset = fullyLoadedSamples P.!! (resetCycle + 10)

      putStrLn "\n=== fullyLoaded During Reset ==="
      putStrLn $ "Before reset (cycle " P.++ show (resetCycle-1) P.++ "): " P.++ show beforeReset
      putStrLn $ "At reset     (cycle " P.++ show resetCycle P.++ "): " P.++ show atReset
      putStrLn $ "After reset  (cycle " P.++ show (resetCycle+1) P.++ "): " P.++ show afterReset
      putStrLn $ "10 cycles after:                   " P.++ show wayAfterReset

      -- The buffer test already passed, so this should be fine
      -- But let's verify the decoder's logic handles it correctly

      -- Simulate decoder's enableAttention with this reset
      let loadTrigger = fromList $ DL.replicate resetCycle False P.++ [True] P.++ DL.repeat False
          enableAttention = mux loadTrigger (pure False) (fullyLoaded <$> bufferSig)
          enableSamples = DL.take totalCycles $ sample enableAttention

          -- Check: enableAttention should be False right after reset
          enableAfterReset = enableSamples P.!! (resetCycle + 1)
          enableWayAfter = enableSamples P.!! (resetCycle + 10)

      putStrLn "\n=== enableAttention After Reset ==="
      putStrLn $ "After reset  (cycle " P.++ show (resetCycle+1) P.++ "): " P.++ show enableAfterReset
      putStrLn $ "10 cycles after:                   " P.++ show enableWayAfter

      -- HERE'S THE POTENTIAL BUG:
      -- If enableAttention becomes True too soon after reset, that's the problem!
      if enableAfterReset || enableWayAfter
        then do
          putStrLn "\n❌ BUG FOUND: enableAttention is True after reset but before new weights loaded!"
          putStrLn "\nROOT CAUSE: The mux logic in Decoder.hs is not sufficient."
          putStrLn "SOLUTION: Need to keep enableAttention False for longer after reset."
          expectationFailure "enableAttention timing violation"
        else
          putStrLn "\n✓ enableAttention correctly stays False after reset"

-- Helpers

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
