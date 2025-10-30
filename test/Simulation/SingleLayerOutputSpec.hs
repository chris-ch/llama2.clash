{-# LANGUAGE ScopedTypeVariables #-}

module Simulation.SingleLayerOutputSpec (spec) where

import           Test.Hspec
import           Clash.Prelude
import qualified Prelude as P
import qualified Data.ByteString.Lazy as BSL

import           LLaMa2.Types.LayerData (Temperature, Seed, Token)
import           LLaMa2.Types.ModelConfig (ModelDimension)
import           LLaMa2.Numeric.Types (FixedPoint)
import           Simulation.DRAMBackedAxiSlave (DRAMConfig (..), buildMemoryFromParams)
import qualified Data.Binary.Get as BG
import qualified Parser
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified LLaMa2.Decoder.Decoder as Decoder
import Simulation.Parameters (DecoderParameters)
import Control.Monad (replicateM)

vectorNorm :: Vec ModelDimension FixedPoint -> Float
vectorNorm v = sqrt $ sum $ P.map (\x -> let f = realToFrac x in f * f) (toList v)

spec :: Spec
spec = do
  describe "Single Layer Output Comparison" $ do
    it "Layer 0 output norm should match expected value of 8.235682" $ do

        modelBinary <- BSL.readFile "data/stories260K.bin"

        let inputToken  = fromList (1 : P.repeat 0) :: Signal System Token
            inputValid  = fromList (True : P.repeat False) :: Signal System Bool
            temperature = pure (0.0 :: Temperature)
            seed        = pure (42 :: Seed)
            powerOn     = pure True
            totalCycles = 10_000

        let 
          params :: DecoderParameters
          !params  = BG.runGet Parser.parseLLaMa2ConfigFile modelBinary
          initMem = buildMemoryFromParams params
          dramCfg = DRAMConfig { readLatency = 1, writeLatency = 0, numBanks = 1 }

          (tok, rdy, ddrMaster', intro) =
            withClockResetEnable systemClockGen resetGen enableGen $
              let
                ddrSlave =
                    DRAM.createDRAMBackedAxiSlaveFromVec dramCfg initMem ddrMaster'
              in Decoder.decoder ddrSlave powerOn params
                      inputToken inputValid temperature seed

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
