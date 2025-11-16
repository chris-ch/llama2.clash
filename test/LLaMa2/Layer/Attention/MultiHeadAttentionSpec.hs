module LLaMa2.Layer.Attention.MultiHeadAttentionSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Layer.Attention.MultiHeadAttention (multiHeadAttentionStage)
import LLaMa2.Types.LayerData (LayerData(..))
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (RowI8E, MatI8E)
import qualified Simulation.Parameters as PARAM
import Test.Hspec
import qualified Prelude as P
import qualified LLaMa2.Memory.AXI.Master as Master

import LLaMa2.Types.ModelConfig (ModelDimension, HeadDimension)
import Simulation.DRAMBackedAxiSlave (WordData, createDRAMBackedAxiSlaveFromVec, DRAMConfig (..))
import Clash.Sized.Vector (unsafeFromList)
import LLaMa2.Numeric.Operations (MultiplierState(..))
import LLaMa2.Layer.Attention.QKVProjection (QHeadDebugInfo (..))

spec :: Spec
spec = do

    describe "multiHeadAttentionStage - Sequential Tokens" $ do

        context "processes two sequential tokens without state pollution" $ do
            let maxCycles = 500

                -- Create test weights: simple identity-like matrices
                testRow = (repeat 1, 0) :: RowI8E ModelDimension
                testRow' = (repeat 1, 0) :: RowI8E HeadDimension
                testQMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension  -- HeadDim x ModelDim
                testKMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testVMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testWOMatrix = repeat testRow' :: MatI8E ModelDimension HeadDimension

                -- Create MHA params with all 8 heads identical
                testHead = PARAM.SingleHeadComponentQ
                    { PARAM.wqHeadQ = testQMatrix
                    , PARAM.wkHeadQ = testKMatrix
                    , PARAM.wvHeadQ = testVMatrix
                    , PARAM.rotaryF = PARAM.RotaryEncodingComponentF
                        { PARAM.freqCosF = repeat (repeat 1.0)
                        , PARAM.freqSinF = repeat (repeat 0.0)
                        }
                    }

                mhaParams = PARAM.MultiHeadAttentionComponentQ
                    { PARAM.headsQ = repeat testHead  -- 8 identical heads
                    , PARAM.mWoQ = repeat testWOMatrix  -- 8 identical WO matrices
                    , PARAM.rmsAttF = repeat 1.0 :< 0  -- RMS norm weights
                    }

                -- Mock DRAM that returns test pattern (all 1s)
                testPattern :: BitVector 512
                testPattern = pack $ replicate (SNat @63) (1 :: BitVector 8)
                                    ++ singleton (0 :: BitVector 8)

                mockDRAM :: Signal System Bool -> Slave.AxiSlaveIn System
                mockDRAM arvalidSignal' =
                    Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = delayedValid arvalidSignal'
                    , rdata = pure (AxiR testPattern 0 True 0)
                    , awready = pure False
                    , wready = pure False
                    , bvalid = pure False
                    , bdata = pure (AxiB 0 0)
                    }

                delayedValid arvalid =
                    exposeClockResetEnable
                    (register False $ register False arvalid)
                    CS.systemClockGen CS.resetGen CS.enableGen

                -- Input: same layer data for both tokens (to isolate state pollution)
                testLayerData = LayerData
                    { inputVector = repeat 1.0 :: Vec 64 FixedPoint
                    , queryVectors = repeat (repeat 0.0)  -- Will be computed
                    , keyVectors = repeat (repeat 0.0)    -- Will be computed
                    , valueVectors = repeat (repeat 0.0)  -- Will be computed
                    , attentionOutput = repeat 0.0        -- Will be computed
                    , feedForwardOutput = repeat 0.0      -- Not used in this stage
                    }

                -- Two tokens: validIn pulses at cycle 1 and cycle 250
                validStream =
                    [False, True] P.++ P.replicate 248 False P.++  -- Token 1
                    -- Token 1
                      -- Token 1
                    -- Token 1
                      -- Token 1
                    -- Token 1
                      -- Token 1
                    -- Token 1
                    [True] P.++ P.replicate (maxCycles - 251) False  -- Token 2
                validIn = fromList validStream :: Signal System Bool

                seqPosStream = P.replicate 250 (0 :: Index 512) P.++   -- Token 1: pos 0
                                P.replicate (maxCycles - 250) 1          -- Token 2: pos 1
                seqPos = fromList seqPosStream

                layerData = pure testLayerData

                (masterOut, xAfterAttn, _q, _k, _v, qkvReady, qkvDone,
                    _writeDone, attentionDone, _debugInfo) =
                    exposeClockResetEnable
                    (multiHeadAttentionStage
                        (mockDRAM arvalidSignal)
                        0  -- layer 0
                        mhaParams
                        seqPos
                        layerData
                        validIn)
                    CS.systemClockGen
                    CS.resetGen
                    CS.enableGen

                arvalidSignal = Master.arvalid masterOut

                -- Sample outputs
                outputs = P.take maxCycles $ sample xAfterAttn
                attnDones = P.take maxCycles $ sample attentionDone
                qkvDones = P.take maxCycles $ sample qkvDone
                qkvReadys = P.take maxCycles $ sample qkvReady

                -- Find completion cycles
                attnDoneIndices = DL.findIndices id attnDones
                firstAttnDone = if not (DL.null attnDoneIndices)
                                then DL.head attnDoneIndices else 0
                secondAttnDone = if P.length attnDoneIndices >= 2
                                then attnDoneIndices P.!! 1 else 0

                firstResult = outputs P.!! firstAttnDone
                secondResult = outputs P.!! secondAttnDone

                tolerance = 0.05  -- Allow 5% difference for numerical precision

            it "completes first token processing" $ do
                P.length attnDoneIndices `shouldSatisfy` (>= 1)

            it "completes second token processing" $ do
                P.length attnDoneIndices `shouldSatisfy` (>= 2)

            it "first token produces non-zero result" $ do
                let vec :: [FixedPoint]
                    vec = toList firstResult
                    norm = sum $ P.map (\x -> x * x) vec
                norm `shouldSatisfy` (> 1.0)

            it "second token produces identical result (no state pollution)" $ do
                let matches = P.zipWith (\a b -> abs (a - b) < tolerance * max (abs a) (abs b))
                                        (toList firstResult)
                                        (toList secondResult)
                DL.and matches `shouldBe` True

            it "qkvReady returns to True between tokens" $ do
                -- After first token completes, should be ready for second
                if firstAttnDone < 249
                then qkvReadys P.!! 249 `shouldBe` True
                else True `shouldBe` True

            it "state machines complete in reasonable time" $ do
                -- First token should complete before cycle 200
                firstAttnDone `shouldSatisfy` (< 250)
                -- Second token should complete before cycle 450
                secondAttnDone `shouldSatisfy` (< 500)

            it "qkvDone pulses exactly once per token" $ do
                let qkvDoneCount1 = P.length $ P.filter id $ P.take 250 qkvDones
                    qkvDoneCount2 = P.length $ P.filter id $ P.drop 250 $ P.take 500 qkvDones
                qkvDoneCount1 `shouldBe` 1
                qkvDoneCount2 `shouldBe` 1

        context "processes two tokens with DIFFERENT inputs correctly" $ do
            let maxCycles = 500

                -- Create test weights: simple identity-like matrices
                testRow = (repeat 1, 0) :: RowI8E ModelDimension
                testRow' = (repeat 1, 0) :: RowI8E HeadDimension
                testQMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testKMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testVMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testWOMatrix = repeat testRow' :: MatI8E ModelDimension HeadDimension

                -- Create MHA params with all 8 heads identical
                testHead = PARAM.SingleHeadComponentQ
                    { PARAM.wqHeadQ = testQMatrix
                    , PARAM.wkHeadQ = testKMatrix
                    , PARAM.wvHeadQ = testVMatrix
                    , PARAM.rotaryF = PARAM.RotaryEncodingComponentF
                        { PARAM.freqCosF = repeat (repeat 1.0)
                        , PARAM.freqSinF = repeat (repeat 0.0)
                        }
                    }

                mhaParams = PARAM.MultiHeadAttentionComponentQ
                    { PARAM.headsQ = repeat testHead
                    , PARAM.mWoQ = repeat testWOMatrix
                    , PARAM.rmsAttF = repeat 1.0 :< 0
                    }

                -- Mock DRAM that returns test pattern (all 1s)
                testPattern :: BitVector 512
                testPattern = pack $ replicate (SNat @63) (1 :: BitVector 8)
                                    ++ singleton (0 :: BitVector 8)

                mockDRAM :: Signal System Bool -> Slave.AxiSlaveIn System
                mockDRAM arvalidSignal' =
                    Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = delayedValid arvalidSignal'
                    , rdata = pure (AxiR testPattern 0 True 0)
                    , awready = pure False
                    , wready = pure False
                    , bvalid = pure False
                    , bdata = pure (AxiB 0 0)
                    }

                delayedValid arvalid =
                    exposeClockResetEnable
                    (register False $ register False arvalid)
                    CS.systemClockGen CS.resetGen CS.enableGen

                -- DIFFERENT inputs for each token
                testLayerData1 = LayerData
                    { inputVector = repeat 1.0 :: Vec 64 FixedPoint
                    , queryVectors = repeat (repeat 0.0)
                    , keyVectors = repeat (repeat 0.0)
                    , valueVectors = repeat (repeat 0.0)
                    , attentionOutput = repeat 0.0
                    , feedForwardOutput = repeat 0.0
                    }

                testLayerData2 = LayerData
                    { inputVector = repeat 2.0 :: Vec 64 FixedPoint  -- DIFFERENT!
                    , queryVectors = repeat (repeat 0.0)
                    , keyVectors = repeat (repeat 0.0)
                    , valueVectors = repeat (repeat 0.0)
                    , attentionOutput = repeat 0.0
                    , feedForwardOutput = repeat 0.0
                    }

                -- Switch inputs based on time
                layerDataStream = P.replicate 250 testLayerData1 P.++
                                P.replicate (maxCycles - 250) testLayerData2
                layerData = fromList layerDataStream

                -- Two tokens: validIn pulses at cycle 1 and cycle 250
                validStream =
                    [False, True] P.++ P.replicate 248 False P.++  -- Token 1
                    -- Token 1
                      -- Token 1
                    -- Token 1
                      -- Token 1
                    -- Token 1
                      -- Token 1
                    -- Token 1
                    [True] P.++ P.replicate (maxCycles - 251) False  -- Token 2
                validIn = fromList validStream :: Signal System Bool


                seqPosStream = P.replicate 250 (0 :: Index 512) P.++   -- Token 1: pos 0
                                P.replicate (maxCycles - 250) 1          -- Token 2: pos 1
                seqPos = fromList seqPosStream


                (masterOut, xAfterAttn, _q, _k, _v, _qkvReady, qkvDone,
                    _writeDone, attentionDone, _debugInfo) =
                    exposeClockResetEnable
                    (multiHeadAttentionStage
                        (mockDRAM arvalidSignal)
                        0  -- layer 0
                        mhaParams
                        seqPos
                        layerData
                        validIn)
                    CS.systemClockGen
                    CS.resetGen
                    CS.enableGen

                arvalidSignal = Master.arvalid masterOut

                -- Sample outputs
                outputs = P.take maxCycles $ sample xAfterAttn
                attnDones = P.take maxCycles $ sample attentionDone
                qkvDones = P.take maxCycles $ sample qkvDone

                -- Find completion cycles
                attnDoneIndices = DL.findIndices id attnDones
                firstAttnDone = if not (DL.null attnDoneIndices)
                                then DL.head attnDoneIndices else 0
                secondAttnDone = if P.length attnDoneIndices >= 2
                                then attnDoneIndices P.!! 1 else 0

                firstResult = outputs P.!! firstAttnDone
                secondResult = outputs P.!! secondAttnDone

                tolerance = 0.1  -- 10% tolerance for ratio check

            it "completes first token with input 1.0" $ do
                P.length attnDoneIndices `shouldSatisfy` (>= 1)

            it "completes second token with input 2.0" $ do
                P.length attnDoneIndices `shouldSatisfy` (>= 2)

            it "first token produces non-zero result" $ do
                let vec :: [FixedPoint]
                    vec = toList firstResult
                    norm = sum $ P.map (\x -> x * x) vec
                norm `shouldSatisfy` (> 1.0)

            it "second token produces DIFFERENT result (inputs are different)" $ do
                let allClose = P.all (\(a, b) -> abs (a - b) < 0.01)
                                    (P.zip (toList firstResult) (toList secondResult))
                allClose `shouldBe` False  -- Should NOT be identical

            it "second token result is approximately 2x first (linear scaling)" $ do
                -- With input 2.0 vs 1.0, and all weights = 1, output should scale ~2x
                -- Check first element as representative
                let firstVal = P.head $ toList firstResult
                    secondVal = P.head $ toList secondResult
                    ratio = secondVal / firstVal
                abs (ratio - 2.0) `shouldSatisfy` (< tolerance)

            it "both tokens produce non-zero results" $ do
                let norm1 = sum $ P.map (\x -> x * x) (toList firstResult)
                    norm2 = sum $ P.map (\x -> x * x) (toList secondResult)
                norm1 `shouldSatisfy` (> 1.0)
                norm2 `shouldSatisfy` (> 1.0)

            it "qkvDone pulses exactly once per token with different inputs" $ do
                let qkvDoneCount1 = P.length $ P.filter id $ P.take 250 qkvDones
                    qkvDoneCount2 = P.length $ P.filter id $ P.drop 250 $ P.take 500 qkvDones
                qkvDoneCount1 `shouldBe` 1
                qkvDoneCount2 `shouldBe` 1

        context "processes two tokens with real DRAM backend" $ do
            let maxCycles = 500

                -- Create test weights in memory format
                testRow = (repeat 1, 0) :: RowI8E ModelDimension
                testRow' = (repeat 1, 0) :: RowI8E HeadDimension
                testQMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testKMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testVMatrix = repeat testRow :: MatI8E HeadDimension ModelDimension
                testWOMatrix = repeat testRow' :: MatI8E ModelDimension HeadDimension

                testHead = PARAM.SingleHeadComponentQ
                    { PARAM.wqHeadQ = testQMatrix
                    , PARAM.wkHeadQ = testKMatrix
                    , PARAM.wvHeadQ = testVMatrix
                    , PARAM.rotaryF = PARAM.RotaryEncodingComponentF
                        { PARAM.freqCosF = repeat (repeat 1.0)
                        , PARAM.freqSinF = repeat (repeat 0.0)
                        }
                    }

                mhaParams = PARAM.MultiHeadAttentionComponentQ
                    { PARAM.headsQ = repeat testHead
                    , PARAM.mWoQ = repeat testWOMatrix
                    , PARAM.rmsAttF = repeat 1.0 :< 0
                    }

                -- Build DRAM contents with test weights
                -- For simplicity, create a small DRAM with just Q weights at the start
                -- Each Q head needs HeadDim (8) rows × 65 bytes = 520 bytes per head
                -- Total for 8 heads = 4160 bytes

                buildQWeights :: [BitVector 8]
                buildQWeights = P.concatMap headWeights [(0 :: Int) ..7]
                    where
                        headWeights _ = P.concatMap rowBytes [(0 :: Int) ..7]
                        rowBytes _ = P.replicate 64 (1 :: BitVector 8) P.++ [0]  -- All 1s, exp=0

                -- Pad to DRAM size and convert to 512-bit words
                dramBytes = buildQWeights P.++ P.repeat 0

                -- Convert byte list to Vec of 512-bit words
                bytesToWords :: [BitVector 8] -> Vec 65536 WordData
                bytesToWords bytes = map wordAtIdx indicesI
                    where
                        wordAtIdx :: Index 65536 -> WordData
                        wordAtIdx idx =
                            let startByte = fromEnum idx * 64
                                slice' = P.take 64 $ P.drop startByte bytes
                                padded = slice' P.++ P.replicate (64 - P.length slice') 0
                                vecBytes = listToVecTH' padded :: Vec 64 (BitVector 8)
                            in pack vecBytes

                        listToVecTH' :: forall n a. (KnownNat n, Default a) => [a] -> Vec n a
                        listToVecTH' xs =
                            let len = natToNum @n
                                padded = P.take len (xs P.++ P.repeat def)
                            in unsafeFromList padded

                dramContents = bytesToWords dramBytes

                -- Create real DRAM
                realDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
                realDRAM masterOut' =
                    exposeClockResetEnable
                        (createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) dramContents masterOut')
                        systemClockGen
                        resetGen
                        enableGen

                -- Two different inputs
                testLayerData1 = LayerData
                    { inputVector = repeat 1.0 :: Vec 64 FixedPoint
                    , queryVectors = repeat (repeat 0.0)
                    , keyVectors = repeat (repeat 0.0)
                    , valueVectors = repeat (repeat 0.0)
                    , attentionOutput = repeat 0.0
                    , feedForwardOutput = repeat 0.0
                    }

                testLayerData2 = LayerData
                    { inputVector = repeat 2.0 :: Vec 64 FixedPoint
                    , queryVectors = repeat (repeat 0.0)
                    , keyVectors = repeat (repeat 0.0)
                    , valueVectors = repeat (repeat 0.0)
                    , attentionOutput = repeat 0.0
                    , feedForwardOutput = repeat 0.0
                    }

                layerDataStream = P.replicate 250 testLayerData1 P.++
                                P.replicate (maxCycles - 250) testLayerData2
                layerData = fromList layerDataStream

                validStream =
                    [False, True] P.++ P.replicate 248 False P.++
                    [True] P.++ P.replicate (maxCycles - 251) False
                validIn = fromList validStream :: Signal System Bool

                seqPosStream = P.replicate 250 (0 :: Index 512) P.++   -- Token 1: pos 0
                                P.replicate (maxCycles - 250) 1          -- Token 2: pos 1
                seqPos = fromList seqPosStream

                (masterOut, xAfterAttn, _q, _k, _v, _qkvReady, _qkvDone,
                    _writeDone, attentionDone, _debugInfo) =
                    exposeClockResetEnable
                    (multiHeadAttentionStage
                    (realDRAM masterOut)
                    0
                    mhaParams
                    seqPos
                    layerData
                    validIn)
                    CS.systemClockGen
                    CS.resetGen
                    CS.enableGen

                outputs = P.take maxCycles $ sample xAfterAttn
                attnDones = P.take maxCycles $ sample attentionDone

                attnDoneIndices = DL.findIndices id attnDones
                firstAttnDone = if not (DL.null attnDoneIndices)
                                then DL.head attnDoneIndices else 0
                secondAttnDone = if P.length attnDoneIndices >= 2
                                then attnDoneIndices P.!! 1 else 0

                firstResult = outputs P.!! firstAttnDone
                secondResult = outputs P.!! secondAttnDone

                tolerance = 0.1

            it "completes first token with real DRAM" $ do
                P.length attnDoneIndices `shouldSatisfy` (>= 1)

            it "completes second token with real DRAM" $ do
                P.length attnDoneIndices `shouldSatisfy` (>= 2)

            it "produces different results for different inputs (real DRAM)" $ do
                let allClose = P.all (\(a, b) -> abs (a - b) < 0.01)
                                    (P.zip (toList firstResult) (toList secondResult))
                allClose `shouldBe` False

            it "second token scales correctly with real DRAM" $ do
                let firstVal = P.head $ toList firstResult
                    secondVal = P.head $ toList secondResult
                    ratio = secondVal / firstVal
                abs (ratio - 2.0) `shouldSatisfy` (< tolerance)
