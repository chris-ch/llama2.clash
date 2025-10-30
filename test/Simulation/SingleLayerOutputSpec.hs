{-# LANGUAGE ScopedTypeVariables #-}

module Simulation.SingleLayerOutputSpec (spec) where

import           Test.Hspec
import           Clash.Prelude
import qualified Prelude as P

import           LLaMa2.Types.LayerData (Temperature, Seed, Token, LayerData (..))
import           LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified LLaMa2.Decoder.Decoder as Decoder
import Simulation.Parameters (DecoderParameters)
import qualified Simulation.ParamsPlaceholder as PARAM
import Control.Monad (unless)
import qualified System.Directory as DIR

vectorNorm :: Vec a FixedPoint -> Float
vectorNorm v = sqrt $ sum $ P.map (\x -> let f = realToFrac x in f * f) (toList v)

spec :: Spec
spec = do
  describe "Single Layer Output Comparison" $ do
    it "Layer 0 data and output norms should match expected values" $ do
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
            totalCycles = 1_600

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

        -- -------------------------------------------------------------------
        -- Sample *all* internal signals we need
        -- -------------------------------------------------------------------
        let sampledLayerIdx   = sampleN totalCycles (Decoder.layerIndex intro)
            sampledFFNDone    = sampleN totalCycles (Decoder.ffnDone intro)
            sampledLayerOut   = sampleN totalCycles (Decoder.layerOutput intro)
            sampledLayerData  = sampleN totalCycles (Decoder.layerData intro)

        -- -------------------------------------------------------------------
        -- Pair everything with a cycle index
        -- -------------------------------------------------------------------
        let
          cycles = [0..totalCycles-1] :: [Int]
          zipped = zipWith4' (,,,)
                              cycles
                              sampledLayerIdx
                              sampledFFNDone
                              (P.zip sampledLayerOut sampledLayerData)

        -- -------------------------------------------------------------------
        -- Keep only cycles where layer 0 has just finished its FFN
        -- -------------------------------------------------------------------
        let layer0Completions =
              [ (i, output, layerData)
              | (i, layerIdx, ffnDone, (output, layerData)) <- zipped
              , layerIdx == 0 && ffnDone ]

        case layer0Completions of
          [] -> expectationFailure "Layer 0 never completed"

          ((cycle', output, layerData) : _) -> do
            -- ---------------------------------------------------------------
            -- 1. Output-norm check (kept unchanged)
            -- ---------------------------------------------------------------
            let outNorm = vectorNorm output
            putStrLn "\n=== Layer 0 Completion ==="
            putStrLn $ "Cycle: " P.++ show cycle'
            abs (outNorm - 8.24) `shouldSatisfy` (< 0.01)

            -- ---------------------------------------------------------------
            -- 2. New layerData checks
            -- ---------------------------------------------------------------
            let LayerData {  attentionOutput
                          , feedForwardOutput
                          } = layerData
{-                           
            inputVector :: Vec ModelDimension FixedPoint,
            queryVectors :: Vec NumQueryHeads (Vec HeadDimension FixedPoint)
            keyVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
            valueVectors :: Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
            attentionOutput :: Vec ModelDimension FixedPoint
            feedForwardOutput :: Vec ModelDimension FixedPoint
-}
            let attnNorm   = vectorNorm attentionOutput   :: Float
                ffnOutNorm = vectorNorm feedForwardOutput    :: Float

            putStrLn "\n--- layerData field norms (first completion) ---"
            putStrLn $ "attentionOutput norm : " P.++ show attnNorm
            putStrLn $ "feedForwardOutput   norm : " P.++ show ffnOutNorm

            -- Reference values obtained from a golden run
            abs (attnNorm   - 9.43) `shouldSatisfy` (< 0.01)
            abs (ffnOutNorm -  8.24) `shouldSatisfy` (< 0.01)

zipWith4' :: (a -> b -> c -> d -> e) -> [a] -> [b] -> [c] -> [d] -> [e]
zipWith4' _ []      _      _      _      = []
zipWith4' _ _      []      _      _      = []
zipWith4' _ _      _      []      _      = []
zipWith4' _ _      _      _      []      = []
zipWith4' f (x:xs) (y:ys) (z:zs) (w:ws) =
    f x y z w : zipWith4' f xs ys zs ws
