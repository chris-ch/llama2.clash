module LLaMa2.Decoder.MultiTokenSpec (spec) where

import Clash.Prelude
import Test.Hspec
import qualified Prelude as P
import qualified Data.List as DL

import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer
import LLaMa2.Memory.LayerAddressing (WeightAddress(..), WeightMatrixType(..))
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Types.ModelConfig
  ( ModelDimension
  , HeadDimension
  , NumQueryHeads
  , NumKeyValueHeads
  )
import Data.Maybe (fromJust)

spec :: Spec
spec = do
  describe "Multi-token weight buffer behavior" $ do
    it "buffer should stay fullyLoaded=True after first load completes" $ do
      -- This tests what happens AFTER the first successful load
      -- Hypothesis: fullyLoaded might be clearing when it shouldn't

      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          totalCycles = lastWriteCycle + 10000  -- Much longer simulation

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          loadedFlags = DL.map fullyLoaded samples

          -- Find when fullyLoaded first becomes True
          firstLoadedIdx = fromJust $ DL.findIndex id loadedFlags

          -- Check if it STAYS True for the next 5000 cycles
          subsequentFlags = DL.take 5000 $ DL.drop (firstLoadedIdx + 1) loadedFlags
          allStayTrue = DL.and subsequentFlags

      -- CRITICAL: Once loaded, should STAY loaded (no reset)
      allStayTrue `shouldBe` True

      P.putStrLn $ "\nFirst loaded at cycle: " P.++ show firstLoadedIdx
      P.putStrLn $ "Stayed loaded for 5000 cycles: " P.++ show allStayTrue

      -- If this fails, fullyLoaded is clearing unexpectedly

    it "buffer should clear fullyLoaded ONLY when reset fires" $ do
      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          resetCycle = lastWriteCycle + 100
          totalCycles = resetCycle + 200

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes

          -- Reset pulse at resetCycle
          reset = fromList $ DL.replicate resetCycle False P.++ [True] P.++ DL.repeat False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          loadedFlags = DL.map fullyLoaded samples

          -- Check states
          beforeReset = loadedFlags DL.!! (resetCycle - 1)
          afterReset = loadedFlags DL.!! (resetCycle + 1)

      beforeReset `shouldBe` True   -- Should be loaded before reset
      afterReset `shouldBe` False   -- Should clear after reset

      P.putStrLn $ "\nBefore reset: " P.++ show beforeReset
      P.putStrLn $ "After reset: " P.++ show afterReset

    it "detects if fullyLoaded glitches during normal operation" $ do
      -- Check for any unexpected transitions from True→False
      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          totalCycles = lastWriteCycle + 5000

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          reset = pure False  -- NO RESET

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          samples = DL.take totalCycles $ sample bufferSig
          loadedFlags = DL.map fullyLoaded samples

          -- Find first True
          firstTrue = fromJust $ DL.findIndex id loadedFlags

          -- Look for any False after first True (with no reset)
          flagsAfterLoad = DL.drop (firstTrue + 1) loadedFlags
          anyGlitch = DL.any not flagsAfterLoad

      anyGlitch `shouldBe` False  -- Should never glitch

      if anyGlitch
        then do
          let firstGlitchIdx = firstTrue + 1 + fromJust (DL.findIndex not flagsAfterLoad)
          P.putStrLn $ "\n✗ GLITCH DETECTED at cycle " P.++ show firstGlitchIdx
          P.putStrLn "This would cause incorrect tokens!"
        else
          P.putStrLn "\n✓ No glitches - fullyLoaded stays stable"


-- Helper functions (same as before)
makeSyntheticRow :: Index HeadDimension -> RowI8E ModelDimension
makeSyntheticRow rowIdx =
  ( imap (\i _ -> fromIntegral (fromEnum rowIdx * 10 + fromEnum i)) (repeat (0 :: Signed 8))
  , fromIntegral (fromEnum rowIdx)
  )

generateWriteSequence :: Int -> [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
generateWriteSequence startCycle = qWrites P.++ kWrites P.++ vWrites
 where
  numQHeads  = fromInteger (natToNum @NumQueryHeads) :: Int
  numKVHeads = fromInteger (natToNum @NumKeyValueHeads) :: Int
  hdDim      = fromInteger (natToNum @HeadDimension) :: Int

  qWrites =
    [ (writeCycle, True,
        WeightAddress rowIdx QMatrix (fromIntegral hd),
        makeSyntheticRow rowIdx)
    | hd <- [0 .. numQHeads - 1]
    , rowIdx <- allIndices
    , let writeCycle = startCycle + hd * hdDim + fromEnum rowIdx
    ]
   where allIndices = [minBound .. maxBound] :: [Index HeadDimension]

  kWrites =
    [ (writeCycle, True,
        WeightAddress rowIdx KMatrix (fromIntegral hd),
        makeSyntheticRow rowIdx)
    | let baseOffset = P.length qWrites
    , hd <- [0 .. numKVHeads - 1]
    , rowIdx <- allIndices
    , let writeCycle = startCycle + baseOffset + hd * hdDim + fromEnum rowIdx
    ]
   where allIndices = [minBound .. maxBound] :: [Index HeadDimension]

  vWrites =
    [ (writeCycle, True,
        WeightAddress rowIdx VMatrix (fromIntegral hd),
        makeSyntheticRow rowIdx)
    | let baseOffset = P.length qWrites + P.length kWrites
    , hd <- [0 .. numKVHeads - 1]
    , rowIdx <- allIndices
    , let writeCycle = startCycle + baseOffset + hd * hdDim + fromEnum rowIdx
    ]
   where allIndices = [minBound .. maxBound] :: [Index HeadDimension]

createSignalsFromSequence ::
  Int -> [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
  -> ( Signal System Bool
     , Signal System WeightAddress
     , Signal System (RowI8E ModelDimension)
     , Signal System Bool
     )
createSignalsFromSequence totalCycles writes =
  (streamValidSig, addrSig, rowSig, allDoneSig)
  where
    writeMap = DL.map (\(c, v, a, r) -> (c, (v, a, r))) writes
    defaultAddr = WeightAddress 0 QMatrix 0
    defaultRow = (repeat 0, 0)

    streamValidSig = fromList
      [ P.maybe False (\(v,_,_) -> v) (DL.lookup c writeMap)
      | c <- [0 .. totalCycles - 1]
      ]

    addrSig = fromList
      [ P.maybe defaultAddr (\(_,a,_) -> a) (DL.lookup c writeMap)
      | c <- [0 .. totalCycles - 1]
      ]

    rowSig = fromList
      [ P.maybe defaultRow (\(_,_,r) -> r) (DL.lookup c writeMap)
      | c <- [0 .. totalCycles - 1]
      ]

    allDoneSig = fromList
      [ case DL.lookup c writeMap of
          Just (True, addr, _) -> isLastVWrite addr
          _ -> False
      | c <- [0 .. totalCycles - 1]
      ]

isLastVWrite :: WeightAddress -> Bool
isLastVWrite WeightAddress{..} =
  matrixType == VMatrix
  && headIndex == fromInteger (natToNum @NumKeyValueHeads - 1)
  && rowIndex == maxBound
