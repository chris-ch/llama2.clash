module LLaMa2.Layer.Attention.WeightLoaderSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import LLaMa2.Layer.Attention.WeightLoader (weightLoader)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E (..))
import LLaMa2.Numeric.Types (Mantissa)
import LLaMa2.Types.ModelConfig
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import qualified Simulation.Parameters as PARAM
import Test.Hspec
import qualified Prelude as P
import qualified Data.List as DL

-- Helper to create test parameters
createTestParams :: MatI8E HeadDimension ModelDimension -> PARAM.DecoderParameters
createTestParams testMatrix =
  PARAM.DecoderParameters
    { PARAM.modelEmbedding =
        PARAM.EmbeddingComponentQ
          { PARAM.vocabularyQ = repeat testRow0,
            PARAM.rmsFinalWeightF = repeat 1.0
          },
      PARAM.modelLayers = repeat layerParams
    }
  where
    testRow0 = head testMatrix
    mockRotary =
      PARAM.RotaryEncodingComponentF
        { PARAM.freqCosF = repeat (repeat 1.0),
          PARAM.freqSinF = repeat (repeat 0.0)
        }
    mockHeadParams =
      PARAM.SingleHeadComponentQ
        { PARAM.wqHeadQ = testMatrix,
          PARAM.wkHeadQ = testMatrix,
          PARAM.wvHeadQ = testMatrix,
          PARAM.rotaryF = mockRotary
        }
    testRow' = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
    testWOMatrix = repeat testRow'
    mhaParams =
      PARAM.MultiHeadAttentionComponentQ
        { PARAM.headsQ = repeat mockHeadParams,
          PARAM.mWoQ = repeat testWOMatrix,
          PARAM.rmsAttF = repeat 1.0 :< 0
        }
    ffnW1 = repeat testRow0
    ffnW2 = repeat RowI8E {rowMantissas = repeat 1, rowExponent = 0}
    ffnW3 = repeat testRow0
    ffnParams =
      PARAM.FeedForwardNetworkComponentQ
        { PARAM.fW1Q = ffnW1,
          PARAM.fW2Q = ffnW2,
          PARAM.fW3Q = ffnW3,
          PARAM.fRMSFfnF = repeat 1.0 :< 0
        }
    layerParams =
      PARAM.TransformerLayerComponent
        { PARAM.multiHeadAttention = mhaParams,
          PARAM.feedforwardNetwork = ffnParams
        }

spec :: Spec
spec = do
  
  -- The new test case for the hardcoded path
  describe "WeightLoader - Hardcoded Path Validation (hcRow)" $ do
    it "returns hardcoded weight instantly upon request" $ do
      -- Setup
      let maxCycles = 20
          -- Request rows 0 through 7 sequentially
          rowReqs = fromList $ [0..7] P.++ P.replicate (maxCycles - 8) 0
          -- The hardcoded path does not need valid/ready signals for correctness, so we keep them simple
          rowReqValids = pure True
          downstreamReadySig = pure True
          
          -- Test matrix: Row 0 mantissa 1, Row 1 mantissa 2, ..., Row 7 mantissa 8
          rows = P.map (\i -> RowI8E {rowMantissas = repeat i, rowExponent = 0}) [1..8]
          testMatrix = P.head rows :> rows P.!! 1 :> rows P.!! 2 :> rows P.!! 3 :>
                      rows P.!! 4 :> rows P.!! 5 :> rows P.!! 6 :> rows P.!! 7 :> Nil
          params = createTestParams testMatrix

          -- Dummy AXI setup since we ignore it
          stubSlaveIn =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen

          -- ASSUMPTION: The weightLoader component is configured to return hcRow for this test
          (masterOut, weightOutput, _, _) =
            exposeClockResetEnable
              (weightLoader stubSlaveIn 0 0 rowReqs rowReqValids downstreamReadySig params)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen

          -- Sample the output weights
          mantissas :: Signal System (Vec ModelDimension Mantissa)
          mantissas = rowMantissas <$> weightOutput
          mantissa :: Signal System Mantissa
          mantissa = (!! 0) <$> mantissas
          mant0s :: [Mantissa]
          mant0s = sampleN maxCycles mantissa

      -- The output should be delayed by exactly 1 cycle (the top-level register)
      -- Cycle 0: Reset value (0)
      -- Cycle 1: Row 0 requested at C0 (mantissa 1)
      -- Cycle 2: Row 1 requested at C1 (mantissa 2)
      let expectedMants = [1, 2, 3, 4, 5, 6, 7, 8] P.++ P.replicate 12 1

      P.putStrLn "\n=== HARDCODED WEIGHT VALIDATION ==="
      P.putStrLn $ "Actual Mantissas (C0-C8): " P.++ show (P.take 9 mant0s)
      P.putStrLn $ "Expected Mantissas (C0-C8): " P.++ show (P.take 9 expectedMants)

      -- Check cycles 1 through 8 (the actual data transfers)
      P.take 9 mant0s `shouldBe` P.take 9 expectedMants

  --------------------------------------------------------------------------------
  -- The existing test for the DRAM path (renamed for clarity)
  --------------------------------------------------------------------------------
  describe "WeightLoader - DRAM Path Consistency (dramRow)" $ do
    it "loads identical weights for consecutive tokens" $ do
      let maxCycles = 200
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          rows = P.map (\i -> RowI8E {rowMantissas = repeat i, rowExponent = 0}) [1..8]
          testMatrix = P.head rows :> rows P.!! 1 :> rows P.!! 2 :> rows P.!! 3 :>
                      rows P.!! 4 :> rows P.!! 5 :> rows P.!! 6 :> rows P.!! 7 :> Nil
          params = createTestParams testMatrix

          rowReqs :: Signal System (Index HeadDimension)
          rowReqs = fromList $
            [0] P.++ P.replicate 10 0 P.++
            P.replicate 10 1 P.++
            P.replicate 10 2 P.++
            P.replicate 10 3 P.++
            P.replicate 10 4 P.++
            P.replicate 10 5 P.++
            P.replicate 10 6 P.++
            P.replicate 10 7 P.++
            P.replicate 20 0 P.++
            P.replicate 10 0 P.++
            P.replicate 10 1 P.++
            P.replicate 10 2 P.++
            P.replicate 10 3 P.++
            P.replicate 10 4 P.++
            P.replicate 10 5 P.++
            P.replicate 10 6 P.++
            P.replicate 10 7 P.++
            P.replicate 20 0

          rowReqValids :: Signal System Bool
          rowReqValids = fromList $
            [False, True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            P.replicate 20 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            [True] P.++ P.replicate 9 False P.++
            P.replicate 20 False

          downstreamReadySig = pure True

          stubSlaveIn =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen

          -- ASSUMPTION: The weightLoader component is configured to return dramRow for this test
          (masterOut, dramRow, dramDataValid, dramReady) =
            exposeClockResetEnable
              (weightLoader stubSlaveIn layerIdx headIdx rowReqs rowReqValids downstreamReadySig params)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen

          mant0s = sampleN maxCycles ((!! (0 :: Int)) . rowMantissas <$> dramRow)
          dataValids = sampleN maxCycles dramDataValid
          readys = sampleN maxCycles dramReady
          reqValids = sampleN maxCycles rowReqValids
          reqs = sampleN maxCycles rowReqs

      let acceptedRequests = P.filter (\i -> readys P.!! i && reqValids P.!! i) [0..maxCycles-1]
          acceptedRows = P.map (\i -> (i, reqs P.!! i)) acceptedRequests
      
      P.putStrLn "\n=== ACCEPTED REQUESTS ==="
      P.mapM_ (\(cycle', row) ->
        P.putStrLn $ "  Cycle " P.++ show cycle' P.++ ": Row " P.++ show row P.++ " accepted")
        acceptedRows

      let validCycles = P.filter (dataValids P.!!) [0..maxCycles-1]
          totalValidCount = P.length validCycles
      
      P.putStrLn $ "\n=== DATA VALID CYCLES (Total: " P.++ show totalValidCount P.++ ") ==="
      
      let safeMatches = P.take totalValidCount (P.zip validCycles (P.map snd acceptedRows))
      
      P.mapM_ (\(validIdx, (validCycle, acceptedRow)) ->
        let actualMant = mant0s P.!! validCycle
            expectedMant = fromIntegral (fromEnum acceptedRow + 1) :: Mantissa
        in P.putStrLn $ "  Valid #" P.++ show (validIdx :: Int) P.++
                       " @ Cycle " P.++ show validCycle P.++
                       ": expected row " P.++ show acceptedRow P.++
                       " (mant=" P.++ show expectedMant P.++ ")" P.++
                       ", got mant=" P.++ show actualMant P.++
                       (if actualMant == expectedMant then " ✓" else " ✗ MISMATCH!"))
        (P.zip [0..] safeMatches)

      let token1ValidCount = P.length (P.filter (< 100) validCycles)
          token2ValidCount = P.length (P.filter (>= 100) validCycles)
      
      P.putStrLn "\n=== TOKEN ANALYSIS ==="
      P.putStrLn $ "Token 1: " P.++ show token1ValidCount P.++ " rows completed (expected 8)"
      P.putStrLn $ "Token 2: " P.++ show token2ValidCount P.++ " rows completed (expected 8)"

      if token1ValidCount < 8
        then do
          P.putStrLn $ "✗ CRITICAL: Token 1 incomplete! Only " P.++ show token1ValidCount P.++ "/8 rows"
          expectationFailure $ "Token 1 incomplete: " P.++ show token1ValidCount P.++ "/8 rows"
        else if token2ValidCount < 8
        then do
          P.putStrLn $ "✗ CRITICAL: Token 2 incomplete! Only " P.++ show token2ValidCount P.++ "/8 rows"
          expectationFailure $ "Token 2 incomplete: " P.++ show token2ValidCount P.++ "/8 rows"
        else do
          let token1Valids = P.take 8 validCycles
              token2Valids = P.take 8 (P.drop 8 validCycles)
              token1Mants = P.map (mant0s P.!!) token1Valids
              token2Mants = P.map (mant0s P.!!) token2Valids

          P.putStrLn "\n=== TOKEN COMPARISON ==="
          P.putStrLn $ "Token 1 mantissas: " P.++ show token1Mants
          P.putStrLn $ "Token 2 mantissas: " P.++ show token2Mants
          P.putStrLn "Expected: [1,2,3,4,5,6,7,8] for both"

          if token1Mants == token2Mants && token1Mants == [1,2,3,4,5,6,7,8]
            then P.putStrLn "✓ PASS: Both tokens got correct, identical weights"
            else do
              P.putStrLn "✗ FAIL: Token weights differ or are incorrect!"
              expectationFailure "Weight corruption between tokens detected"

          token1Mants `shouldBe` [1,2,3,4,5,6,7,8]
          token2Mants `shouldBe` [1,2,3,4,5,6,7,8]
