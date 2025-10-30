{-# LANGUAGE ScopedTypeVariables #-}

module Simulation.SingleLayerOutputSpec (spec) where

import           Test.Hspec
import           Clash.Prelude
import qualified Prelude as P

import           LLaMa2.Types.LayerData (Temperature, Seed, Token)
import           LLaMa2.Types.ModelConfig (ModelDimension)
import           LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified LLaMa2.Decoder.Decoder as Decoder
import Simulation.Parameters (DecoderParameters)
import qualified Simulation.ParamsPlaceholder as PARAM
import Control.Monad (unless)
import qualified System.Directory as DIR

vectorNorm :: Vec ModelDimension FixedPoint -> Float
vectorNorm v = sqrt $ sum $ P.map (\x -> let f = realToFrac x in f * f) (toList v)

spec :: Spec
spec = do
  describe "Single Layer Output Comparison" $ do
    it "Layer 0 output norm should match expected value of 8.235682" $ do
        -- Check file exists
        fileExists <- DIR.doesFileExist "./data/stories260K.bin"
        cwd <- DIR.getCurrentDirectory
        putStrLn $ "Test running from: " P.++ cwd
        unless fileExists $ 
          expectationFailure "Model file not found at ./data/stories260K.bin"
      
        let inputToken  = fromList (1 : P.repeat 1) :: Signal System Token
            inputValid  = fromList (True : P.repeat True) :: Signal System Bool
            temperature = pure (0.0 :: Temperature)
            seed        = pure (123 :: Seed)
            powerOn     = pure True
            totalCycles = 2_000

        let 
          params :: DecoderParameters
          params = PARAM.decoderConst

          ddrSlave = exposeClockResetEnable
            (DRAM.createDRAMBackedAxiSlave params ddrMaster)
            systemClockGen
            resetGen
            enableGen

          -- Create the feedback loop: masters drive slaves, slaves feed back to decoder
          (tok, rdy, ddrMaster, intro) =
            exposeClockResetEnable
              (Decoder.decoder ddrSlave powerOn params inputToken inputValid temperature seed)
              systemClockGen
              resetGen
              enableGen

        let sampledLayerIdx = sampleN totalCycles (Decoder.layerIndex intro)
            sampledFFNDone  = sampleN totalCycles (Decoder.ffnDone intro)
            sampledLayerOut = sampleN totalCycles (Decoder.layerOutput intro)

            layer0Completions =
              [ (i, output)
              | (i, (layerIdx, ffnDone, output)) <-
                      P.zip [0 :: Int ..] (P.zip3 sampledLayerIdx sampledFFNDone sampledLayerOut)
              , layerIdx == 0 && ffnDone ]

        case layer0Completions of
          [] -> expectationFailure "Layer 0 never completed"
          ((cycle', output):_) -> do
            let norm = vectorNorm output
            putStrLn "\n=== Layer 0 Completion ==="
            putStrLn $ "Cycle: " P.++ show cycle'
            putStrLn $ "Output norm: " P.++ show norm
            putStrLn   "Expected: 8.235682"
            abs (norm - 8.235682) `shouldSatisfy` (< 0.1)
