module Simulation.ControllerSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P
import qualified LLaMa2.Decoder.SimplifiedSequenceController as Controller
import LLaMa2.Types.ModelConfig (NumLayers)

spec :: Spec
spec = do
  describe "Unified Controller Enable Signals" $ do

    it "should generate correct stage enable sequence with proper register timing" $ do
      let totalCycles = 20

          -- Simulate instant completion of each stage
          qkvDone = pure True
          writeDone = pure True
          attnDone = pure True
          ffnDone = pure True
          classifierDone = pure True

          controller = exposeClockResetEnable
            (Controller.unifiedController qkvDone writeDone attnDone ffnDone classifierDone)
            systemClockGen
            resetGen
            enableGen

          sampledEnableQKV = sampleN totalCycles (Controller.enableQKV controller)
          sampledEnableWriteKV = sampleN totalCycles (Controller.enableWriteKV controller)
          sampledEnableAttend = sampleN totalCycles (Controller.enableAttend controller)
          sampledEnableFFN = sampleN totalCycles (Controller.enableFFN controller)
          sampledLayer = sampleN totalCycles (Controller.currentLayer controller)

      -- With pure True done signals and register initialization:
      -- First stage takes 2 cycles (initialization), then advances every cycle

      -- Cycles 0-1: QKV (layer 0) - takes 2 cycles due to init
      P.head sampledEnableQKV `shouldBe` True
      P.head sampledEnableWriteKV `shouldBe` False
      P.head sampledLayer `shouldBe` 0

      sampledEnableQKV P.!! 1 `shouldBe` True  -- Still QKV
      sampledLayer P.!! 1 `shouldBe` 0

      -- Cycle 2: WriteKV (layer 0) - now advances every cycle
      sampledEnableQKV P.!! 2 `shouldBe` False
      sampledEnableWriteKV P.!! 2 `shouldBe` True
      sampledLayer P.!! 2 `shouldBe` 0

      -- Cycle 3: Attend (layer 0)
      sampledEnableWriteKV P.!! 3 `shouldBe` False
      sampledEnableAttend P.!! 3 `shouldBe` True
      sampledLayer P.!! 3 `shouldBe` 0

      -- Cycle 4: FFN (layer 0)
      sampledEnableAttend P.!! 4 `shouldBe` False
      sampledEnableFFN P.!! 4 `shouldBe` True
      sampledLayer P.!! 4 `shouldBe` 0

      -- Cycle 5: QKV (layer 1) - advances to next layer
      sampledEnableFFN P.!! 5 `shouldBe` False
      sampledEnableQKV P.!! 5 `shouldBe` True
      sampledLayer P.!! 5 `shouldBe` 1

    it "should have exactly one stage enabled at a time" $ do
      let totalCycles = 40
          
          qkvDone = pure True
          writeDone = pure True
          attnDone = pure True
          ffnDone = pure True
          classifierDone = pure True
          
          controller = exposeClockResetEnable
            (Controller.unifiedController qkvDone writeDone attnDone ffnDone classifierDone)
            systemClockGen
            resetGen
            enableGen
          
          sampledEnableQKV = sampleN totalCycles (Controller.enableQKV controller)
          sampledEnableWriteKV = sampleN totalCycles (Controller.enableWriteKV controller)
          sampledEnableAttend = sampleN totalCycles (Controller.enableAttend controller)
          sampledEnableFFN = sampleN totalCycles (Controller.enableFFN controller)
          sampledEnableClassifier = sampleN totalCycles (Controller.enableClassifier controller)  -- ADD THIS
          
          cyclesWithEnables = zip5'  -- Change from zip4' to zip5'
            sampledEnableQKV sampledEnableWriteKV sampledEnableAttend
            sampledEnableFFN sampledEnableClassifier  -- ADD THIS
          
          countActive (q, w, a, f, c) =  -- Add 'c' parameter
            P.length $ P.filter id [q, w, a, f, c]  -- Add 'c' to list
      
      -- Verify exactly one enable signal is active per cycle
      P.all (\enables -> countActive enables == 1) cyclesWithEnables `shouldBe` True

      -- Verify exactly one enable signal is active per cycle
      P.all (\enables -> countActive enables == 1) cyclesWithEnables `shouldBe` True

    it "should eventually reach classifier and generate ready pulse" $ do
      let numLayers = natToNum @NumLayers
          -- With instant completion: 1 cycle init + 4 cycles per layer + classifier
          cyclesPerLayer = 4  -- QKV, WriteKV, Attend, FFN
          cyclesNeeded = 1 + (numLayers * cyclesPerLayer) + 2
          totalCycles = cyclesNeeded + 10

          qkvDone = pure True
          writeDone = pure True
          attnDone = pure True
          ffnDone = pure True
          classifierDone = pure True

          controller = exposeClockResetEnable
            (Controller.unifiedController qkvDone writeDone attnDone ffnDone classifierDone)
            systemClockGen
            resetGen
            enableGen

          sampledReady = sampleN totalCycles (Controller.readyPulse controller)
          sampledSeqPos = sampleN totalCycles (Controller.seqPosition controller)

      -- Should generate ready pulse after classifier completes
      let readyCycles = P.filter (sampledReady P.!!) [0..totalCycles-1]
      P.length readyCycles `shouldSatisfy` (> 0)

      -- After ready pulse, sequence position should eventually increment
      let firstReadyCycle = P.head readyCycles
          checkCycle = P.min (firstReadyCycle + 3) (totalCycles - 1)

      -- Sequence position should be > 0 a few cycles after ready
      if checkCycle < totalCycles
        then (sampledSeqPos P.!! checkCycle) `shouldSatisfy` (> 0)
        else P.putStrLn "Not enough cycles to check sequence advancement"

zip5' :: [a] -> [b] -> [c] -> [d] -> [e] -> [(a, b, c, d, e)]
zip5' (a:as) (b:bs) (c:cs) (d:ds) (e:es) = (a, b, c, d, e) : zip5' as bs cs ds es
zip5' [] _ _ _ _ = []
zip5' _ [] _ _ _ = []
zip5' _ _ [] _ _ = []
zip5' _ _ _ [] _ = []
zip5' _ _ _ _ [] = []

