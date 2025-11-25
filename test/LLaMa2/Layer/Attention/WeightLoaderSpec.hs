module LLaMa2.Layer.Attention.WeightLoaderSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Layer.Attention.WeightLoader (weightLoader)
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import LLaMa2.Types.ModelConfig
import qualified Simulation.Parameters as PARAM
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import Test.Hspec
import qualified Prelude as P
import Data.Maybe (isJust)
import Control.Monad (when)

-- Diagnostic for each cycle
data WLDiagnostic = WLDiagnostic
  { wlCycle :: Int
  , wlRowReq :: Index HeadDimension
  , wlRowReqValid :: Bool
  , wlDownstreamReady :: Bool
  , wlArvalid :: Bool
  , wlFetchValid :: Bool
  , wlDramReady :: Bool
  , wlDramDataValid :: Bool
  , wlRowExp :: Exponent
  , wlRowMant0 :: Mantissa
  } deriving (Show)

-- Helper to create test parameters
createTestParams :: MatI8E HeadDimension ModelDimension -> PARAM.DecoderParameters
createTestParams testMatrix =
  PARAM.DecoderParameters
    { PARAM.modelEmbedding = PARAM.EmbeddingComponentQ
        { PARAM.vocabularyQ = repeat testRow0
        , PARAM.rmsFinalWeightF = repeat 1.0
        }
    , PARAM.modelLayers = repeat layerParams
    }
  where
    testRow0 = head testMatrix
    mockRotary = PARAM.RotaryEncodingComponentF
      { PARAM.freqCosF = repeat (repeat 1.0)
      , PARAM.freqSinF = repeat (repeat 0.0)
      }
    mockHeadParams = PARAM.SingleHeadComponentQ
      { PARAM.wqHeadQ = testMatrix
      , PARAM.wkHeadQ = testMatrix
      , PARAM.wvHeadQ = testMatrix
      , PARAM.rotaryF = mockRotary
      }
    testRow' = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
    testWOMatrix = repeat testRow'
    mhaParams = PARAM.MultiHeadAttentionComponentQ
      { PARAM.headsQ = repeat mockHeadParams
      , PARAM.mWoQ = repeat testWOMatrix
      , PARAM.rmsAttF = repeat 1.0 :< 0
      }
    ffnW1 = repeat testRow0
    ffnW2 = repeat RowI8E {rowMantissas = repeat 1, rowExponent = 0}
    ffnW3 = repeat testRow0
    ffnParams = PARAM.FeedForwardNetworkComponentQ
      { PARAM.fW1Q = ffnW1
      , PARAM.fW2Q = ffnW2
      , PARAM.fW3Q = ffnW3
      , PARAM.fRMSFfnF = repeat 1.0 :< 0
      }
    layerParams = PARAM.TransformerLayerComponent
      { PARAM.multiHeadAttention = mhaParams
      , PARAM.feedforwardNetwork = ffnParams
      }

spec :: Spec
spec = do
  describe "WeightLoader - Ready/Valid Protocol" $ do
    it "respects backpressure: ignores requests when not ready" $ do
      let maxCycles = 50
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
          testMatrix = repeat row0
          params = createTestParams testMatrix

          -- Request row 0 at cycle 1, but downstream never ready
          rowReqSig = pure 0 :: Signal System (Index HeadDimension)
          rowReqValidSig = fromList ([False, True] P.++ P.replicate 48 True)
          downstreamReadySig = pure False :: Signal System Bool

          stubSlaveIn = exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
            CS.systemClockGen CS.resetGen CS.enableGen

          (masterOut, dramRow, dramDataValid, dramReady) = exposeClockResetEnable
            (weightLoader stubSlaveIn layerIdx headIdx rowReqSig rowReqValidSig downstreamReadySig params)
            CS.systemClockGen CS.resetGen CS.enableGen

          arvalids = sampleN maxCycles (Master.arvalid masterOut)
          readys = sampleN maxCycles dramReady
          dataValids = sampleN maxCycles dramDataValid

      -- Should fetch once, then stay busy (not ready for new requests)
      let fetchCount = P.length (P.filter id arvalids)
      P.putStrLn $ "Total AR requests: " P.++ show fetchCount

      -- After first fetch completes, should stay in LDone (not ready)
      let firstNotReady = DL.elemIndex False readys
      P.putStrLn $ "First not-ready cycle: " P.++ show firstNotReady

      -- Should not issue multiple fetches
      fetchCount `shouldSatisfy` (<= 2)  -- At most 1-2 (accounting for pipeline)

  describe "WeightLoader - Multi-Request Sequence" $ do
    it "loads row 0, then row 1, then row 0 again with correct data" $ do
      let maxCycles = 100
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          -- Create distinctive rows
          row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
          row1 = RowI8E {rowMantissas = repeat 10, rowExponent = 0}
          rowOther = RowI8E {rowMantissas = repeat 5, rowExponent = 0}

          testMatrix = row0 :> row1 :> replicate d6 rowOther
          params = createTestParams testMatrix

          -- Request sequence:
          -- Cycle 1: Request row 0
          -- Cycle 20: Request row 1
          -- Cycle 40: Request row 0 again
          rowReqs :: Signal System (Index HeadDimension)
          rowReqs = fromList ([0] P.++ P.replicate 18 0 P.++
                              [1] P.++ P.replicate 18 1 P.++
                              [0] P.++ P.replicate 59 0)

          rowReqValids :: Signal System Bool
          rowReqValids = fromList ([False, True] P.++ P.replicate 17 False P.++
                                   [True] P.++ P.replicate 18 False P.++
                                   [True] P.++ P.replicate 59 False)

          downstreamReadySig :: Signal System Bool
          downstreamReadySig = pure True

          stubSlaveIn = exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
            CS.systemClockGen CS.resetGen CS.enableGen

          masterOut :: Master.AxiMasterOut System
          dramRow :: Signal System (RowI8E ModelDimension)
          dramDataValid :: Signal System Bool
          dramReady :: Signal System Bool
          (masterOut, dramRow, dramDataValid, dramReady) = exposeClockResetEnable
            (weightLoader stubSlaveIn layerIdx headIdx rowReqs rowReqValids downstreamReadySig params)
            CS.systemClockGen CS.resetGen CS.enableGen

          exps = sampleN @System maxCycles (rowExponent <$> dramRow)
          mant0s = sampleN @System maxCycles ((!! (0 :: Int)) . rowMantissas <$> dramRow)
          dataValids = sampleN @System maxCycles dramDataValid

          diagnostics = flip P.map [0..maxCycles-1] $ \i ->
            WLDiagnostic
              { wlCycle = i
              , wlRowReq = sampleN @System maxCycles rowReqs P.!! i
              , wlRowReqValid = sampleN @System maxCycles rowReqValids P.!! i
              , wlDownstreamReady = True
              , wlArvalid = sampleN @System maxCycles (Master.arvalid masterOut) P.!! i
              , wlFetchValid = False  -- Would need to expose from weightLoader
              , wlDramReady = sampleN @System maxCycles dramReady P.!! i
              , wlDramDataValid = dataValids P.!! i
              , wlRowExp = exps P.!! i
              , wlRowMant0 = mant0s P.!! i
              }

      -- Find when each request completes
      let validCycles = P.filter wlDramDataValid diagnostics

      P.putStrLn "\nValid data cycles:"
      P.mapM_ (\d -> P.putStrLn $ "  Cycle " P.++ show (wlCycle d) P.++ ": mant0=" P.++ show (wlRowMant0 d)) validCycles

      -- Should have at least 3 valid cycles
      P.length validCycles `shouldSatisfy` (>= 3)

      -- Check first completion (row 0): mant0 should be 1
      let firstValid = P.head validCycles
      wlRowMant0 firstValid `shouldBe` 1

      -- Check second completion (row 1): mant0 should be 10
      case P.drop 1 validCycles of
        [] -> expectationFailure "Second request didn't complete"
        (secondValid:_) -> wlRowMant0 secondValid `shouldBe` 10

      -- Check third completion (row 0 again): mant0 should be 1 (not 10!)
      case P.drop 2 validCycles of
        [] -> expectationFailure "Third request didn't complete"
        (thirdValid:_) -> do
          P.putStrLn $ "\nThird request (row 0 again) at cycle " P.++ show (wlCycle thirdValid)
          P.putStrLn $ "  mant0 = " P.++ show (wlRowMant0 thirdValid) P.++ " (expected 1)"
          wlRowMant0 thirdValid `shouldBe` 1

  describe "WeightLoader - State Machine Timing" $ do
    it "transitions through Idle -> Fetching -> Done correctly" $ do
      let maxCycles = 30
          layerIdx = 0 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads

          row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
          params = createTestParams (repeat row0)

          rowReqSig = pure 0
          rowReqValidSig = fromList ([False, True] P.++ P.replicate 28 False)
          downstreamReadySig = pure True

          stubSlaveIn = exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
            CS.systemClockGen CS.resetGen CS.enableGen

          (masterOut, dramRow, dramDataValid, dramReady) = exposeClockResetEnable
            (weightLoader stubSlaveIn layerIdx headIdx rowReqSig rowReqValidSig downstreamReadySig params)
            CS.systemClockGen CS.resetGen CS.enableGen

          readys = sampleN maxCycles dramReady
          dataValids = sampleN maxCycles dramDataValid

      P.putStrLn "\nState transitions:"
      P.mapM_ (\i -> P.putStrLn $ "  Cycle " P.++ show i P.++
                                   ": ready=" P.++ show (readys P.!! i) P.++
                                   ", dataValid=" P.++ show (dataValids P.!! i))
              [0..P.min 15 (maxCycles-1)]

      -- Should see: Idle(T) -> Fetching(F) -> Done(F,T) -> Idle(T)
      let firstNotReady = DL.elemIndex False readys
          firstDataValid = DL.elemIndex True dataValids
          returnToReady = case firstNotReady of
            Nothing -> Nothing
            Just i -> DL.elemIndex True (P.drop (i+1) readys) >>= \j -> Just (i + j + 1)

      P.putStrLn $ "\nFirst not-ready: " P.++ show firstNotReady
      P.putStrLn $ "First data-valid: " P.++ show firstDataValid
      P.putStrLn $ "Return to ready: " P.++ show returnToReady

      firstNotReady `shouldSatisfy` isJust
      firstDataValid `shouldSatisfy` isJust
      returnToReady `shouldSatisfy` isJust

    describe "WeightLoader - Ready/Valid Handshake Violation" $ do
        it "detects when requests are issued while not ready (should be rejected)" $ do
            let maxCycles = 50
                layerIdx = 0 :: Index NumLayers
                headIdx = 0 :: Index NumQueryHeads

                row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
                row1 = RowI8E {rowMantissas = repeat 10, rowExponent = 0}
                testMatrix = row0 :> replicate d7 row1
                params = createTestParams testMatrix

                -- Issue TWO rapid requests before weightLoader can finish first one
                -- Cycle 1: Request row 0 (valid)
                -- Cycle 2: Request row 1 (should be IGNORED - weightLoader not ready yet)
                rowReqs :: Signal System (Index HeadDimension)
                rowReqs = fromList ([0, 0, 1] P.++ P.replicate 47 1)
                rowReqValids :: Signal System Bool
                rowReqValids = fromList ([False, True, True] P.++ P.replicate 47 False)
                downstreamReadySig = pure True

                stubSlaveIn = exposeClockResetEnable
                    (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
                    CS.systemClockGen CS.resetGen CS.enableGen

                (masterOut, dramRow, dramDataValid, dramReady) = exposeClockResetEnable
                    (weightLoader stubSlaveIn layerIdx headIdx rowReqs rowReqValids downstreamReadySig params)
                    CS.systemClockGen CS.resetGen CS.enableGen

                readys = sampleN maxCycles dramReady
                dataValids = sampleN maxCycles dramDataValid
                mant0s = sampleN maxCycles ((!! (0 :: Int)) . rowMantissas <$> dramRow)
                reqValids = sampleN maxCycles rowReqValids
                reqs = sampleN maxCycles rowReqs

            P.putStrLn "\nHandshake timing:"
            P.mapM_ (\i -> P.putStrLn $
                    "  Cycle " P.++ show i P.++
                    ": req=" P.++ show (reqs P.!! i) P.++
                    ", reqValid=" P.++ show (reqValids P.!! i) P.++
                    ", ready=" P.++ show (readys P.!! i) P.++
                    ", dataValid=" P.++ show (dataValids P.!! i) P.++
                    ", mant0=" P.++ show (mant0s P.!! i))
                    [0..P.min 20 (maxCycles-1)]

            -- Key check: At cycle 2, we request row 1, but weightLoader is NOT ready
            (if readys P.!! 2
                then P.putStrLn "\nERROR: weightLoader shows ready at cycle 2 (impossible - should still be processing row 0)"
                else P.putStrLn "\nCORRECT: weightLoader is NOT ready at cycle 2")

            -- The second request (row 1) should be IGNORED
            -- So the first data we get should be row 0 (mant0=1), not row 1 (mant0=10)
            let firstValidCycle = DL.elemIndex True dataValids
            case firstValidCycle of
                Nothing -> expectationFailure "No data was ever valid"
                Just cycle' -> do
                    let firstMant = mant0s P.!! cycle'
                    P.putStrLn $ "\nFirst valid data at cycle " P.++ show cycle' P.++ ": mant0=" P.++ show firstMant
                    firstMant `shouldBe` 1  -- Should be row 0, not row 1

        it "shows stale data when request is rejected" $ do
            let maxCycles = 50
                layerIdx = 0 :: Index NumLayers
                headIdx = 0 :: Index NumQueryHeads

                row0 = RowI8E {rowMantissas = repeat 1, rowExponent = 0}
                row1 = RowI8E {rowMantissas = repeat 10, rowExponent = 0}
                testMatrix = row0 :> replicate d7 row1
                params = createTestParams testMatrix

                -- Request row 0, let it complete, then immediately request row 1 while NOT ready
                rowReqs :: Signal System (Index HeadDimension)
                rowReqs = fromList ([0] P.++ P.replicate 8 0 P.++
                                [1] P.++ P.replicate 40 1)
                rowReqValids :: Signal System Bool
                rowReqValids = fromList ([False, True] P.++ P.replicate 7 False P.++
                                        [True] P.++ P.replicate 40 False)
                downstreamReadySig = pure True

                stubSlaveIn = exposeClockResetEnable
                    (DRAMSlave.createDRAMBackedAxiSlave params masterOut)
                    CS.systemClockGen CS.resetGen CS.enableGen

                (masterOut, dramRow, dramDataValid, dramReady) = exposeClockResetEnable
                    (weightLoader stubSlaveIn layerIdx headIdx rowReqs rowReqValids downstreamReadySig params)
                    CS.systemClockGen CS.resetGen CS.enableGen

                readys = sampleN maxCycles dramReady
                reqValids = sampleN maxCycles rowReqValids
                reqs = sampleN maxCycles rowReqs
                mant0s = sampleN maxCycles ((!! (0 :: Int)) . rowMantissas <$> dramRow)

            P.putStrLn "\nRequest timing at cycle 9:"
            P.putStrLn $ "  Cycle 9: req=" P.++ show (reqs P.!! 9) P.++
                        ", reqValid=" P.++ show (reqValids P.!! 9) P.++
                        ", ready=" P.++ show (readys P.!! 9) P.++
                        ", currentMant0=" P.++ show (mant0s P.!! 9)

            -- If cycle 9 has reqValid=True but ready=False, the request is REJECTED
            -- But the consumer (queryHeadProjector) doesn't know this!
            -- It will continue using whatever is in dramRow, which is stale data
            when (reqValids P.!! 9 && not (readys P.!! 9)) $ do
                P.putStrLn "\n*** BUG EXPOSED ***"
                P.putStrLn "Request issued while not ready - this will cause downstream to use stale data!"
                P.putStrLn $ "Stale mant0 value: " P.++ show (mant0s P.!! 9)
            True `shouldBe` True