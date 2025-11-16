module LLaMa2.Layer.Attention.QKVProjectionSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Layer.Attention.QKVProjection (queryHeadProjector)
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.Parameters as PARAM
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Numeric.Quantization (RowI8E, MatI8E)
import LLaMa2.Types.ModelConfig

spec :: Spec
spec = do
  describe "queryHeadProjector - Sequential Transactions" $ do
    context "produces identical results for two sequential identical inputs" $ do
      let maxCycles = 100

          -- Create simple test weights (8x64 matrix for HeadDimension=8, ModelDimension=64)
          -- Each row: mantissas = [1,1,1,...], exponent = 0
          testRow = (repeat 1, 0) :: RowI8E 64
          testMatrix = repeat testRow :: MatI8E 8 64

          -- Mock DRAM that returns our test pattern
          testPattern :: BitVector 512
          testPattern = pack $ replicate (SNat @63) (1 :: BitVector 8) ++ singleton (0 :: BitVector 8)

          mockDRAM :: Signal System Bool -> Slave.AxiSlaveIn System
          mockDRAM arvalidSignal' =
            Slave.AxiSlaveIn
              { arready = pure True
              , rvalid = delayedValid arvalidSignal'  -- 2-cycle latency
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

          -- Input: same vector both times
          inputVec = repeat 1.0 :: Vec 64 FixedPoint

          -- Mock rotary params (identity - no rotation)
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

          -- Create test weights: simple identity-like matrices
          testRow' = (repeat 1, 0) :: RowI8E HeadDimension
          testWOMatrix = repeat testRow' :: MatI8E ModelDimension HeadDimension

          mhaParams = PARAM.MultiHeadAttentionComponentQ
              { PARAM.headsQ = repeat mockHeadParams
              , PARAM.mWoQ = repeat testWOMatrix
              , PARAM.rmsAttF = repeat 1.0 :< 0
              }

          -- FFN weights (all 1s for simplicity)
          ffnW1 = repeat testRow :: MatI8E HiddenDimension ModelDimension
          ffnW2 = repeat (repeat 1, 0) :: MatI8E ModelDimension HiddenDimension
          ffnW3 = repeat testRow :: MatI8E HiddenDimension ModelDimension

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

          params = PARAM.DecoderParameters
              { PARAM.modelEmbedding = PARAM.EmbeddingComponentQ
                  { PARAM.vocabularyQ = repeat testRow :: MatI8E VocabularySize ModelDimension
                  , PARAM.rmsFinalWeightF = repeat 1.0 :: Vec ModelDimension FixedPoint
                  }
              , PARAM.modelLayers = repeat layerParams
              }

          -- Two transactions: validIn pulses at cycle 1 and cycle 50
          validStream =
            [False, True] P.++ P.replicate 48 False P.++  -- First transaction
              -- First transaction
            [True] P.++ P.replicate (maxCycles - 51) False  -- Second transaction
          validIn = fromList validStream :: Signal System Bool

          downStreamReady = pure True :: Signal System Bool
          stepCount = pure 0 :: Signal System (Index 512)
          input = pure inputVec :: Signal System (Vec 64 FixedPoint)

          (masterOut, qOut, validOut, readyOut, _debugInfo) =
            exposeClockResetEnable
              (queryHeadProjector
                (mockDRAM arvalidSignal)
                0  -- layer 0
                0  -- head 0
                validIn
                downStreamReady
                stepCount
                input
                params)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen

          arvalidSignal = Master.arvalid masterOut

          outputs = P.take maxCycles $ sample qOut
          valids = P.take maxCycles $ sample validOut
          readys = P.take maxCycles $ sample readyOut

          validIndices = DL.findIndices id valids
          firstCompletion = if not (DL.null validIndices) then DL.head validIndices else 0
          secondCompletion = if P.length validIndices >= 2 then validIndices P.!! 1 else 0

          firstResult = outputs P.!! firstCompletion
          secondResult = outputs P.!! secondCompletion

          tolerance = 0.01

      it "completes first transaction" $ do
        P.length validIndices `shouldSatisfy` (>= 1)

      it "completes second transaction" $ do
        P.length validIndices `shouldSatisfy` (>= 2)

      it "first transaction produces non-zero result" $ do
        let
            vec :: [FixedPoint]
            vec = toList firstResult
            norm = sum $ P.map (\x -> x * x) vec
        norm `shouldSatisfy` (> 0.1)

      it "second transaction produces identical result (no state pollution)" $ do
        let matches = P.zipWith (\a b -> abs (a - b) < tolerance)
                                (toList firstResult)
                                (toList secondResult)
        DL.and matches `shouldBe` True

      it "returns to ready between transactions" $ do
        if firstCompletion < maxCycles - 1
          then readys P.!! (firstCompletion + 1) `shouldBe` True
          else True `shouldBe` True
