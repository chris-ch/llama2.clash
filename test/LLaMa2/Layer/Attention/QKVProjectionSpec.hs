module LLaMa2.Layer.Attention.QKVProjectionSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Layer.Attention.QKVProjection (queryHeadProjector, QHeadDebugInfo (..))
import LLaMa2.Memory.AXI.Types
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Numeric.Types (FixedPoint, Mantissa)
import qualified Simulation.Parameters as PARAM
import qualified LLaMa2.Numeric.Operations as OPS
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Numeric.Quantization (RowI8E (..), MatI8E)
import LLaMa2.Types.ModelConfig

-- Diagnostic record for cycle-by-cycle comparison
data CycleDiagnostic = CycleDiagnostic
  { cycleNum :: Int
  , rowIdx :: Index HeadDimension
  , state :: OPS.MultiplierState
  , rowReset :: Bool
  , rowEnable :: Bool
  , fetchValid :: Bool
  , firstMant :: Mantissa
  , accumVal :: FixedPoint
  , rowResult :: FixedPoint
  , rowDone :: Bool
  , qOutVec :: Vec HeadDimension FixedPoint
  , currentRow :: RowI8E ModelDimension
  , currentRow' :: RowI8E ModelDimension
  } deriving (Show, Eq)

spec :: Spec
spec = do
  describe "queryHeadProjector - Diagnostic Comparison" $ do
    context "layer 4 (5th layer), second token processing" $ do
      let maxCycles = 50
          layerIdx = 4 :: Index NumLayers
          headIdx = 0 :: Index NumQueryHeads
          
          -- Create realistic test weights
          testRow = RowI8E { rowMantissas = repeat 1, rowExponent = 0} :: RowI8E 64
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

          -- Input vector for "second token"
          inputVec = repeat 1.0 :: Vec 64 FixedPoint

          -- Mock rotary params (identity)
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

          -- Build full params structure
          testRow' = RowI8E { rowMantissas = repeat 1, rowExponent = 0} :: RowI8E HeadDimension
          testWOMatrix = repeat testRow' :: MatI8E ModelDimension HeadDimension

          mhaParams = PARAM.MultiHeadAttentionComponentQ
              { PARAM.headsQ = repeat mockHeadParams
              , PARAM.mWoQ = repeat testWOMatrix
              , PARAM.rmsAttF = repeat 1.0 :< 0
              }

          ffnW1 = repeat testRow :: MatI8E HiddenDimension ModelDimension
          ffnW2 = repeat RowI8E { rowMantissas = repeat 1, rowExponent = 0} :: MatI8E ModelDimension HiddenDimension
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

          -- Single transaction starting at cycle 1
          validIn = fromList ([False, True] P.++ P.replicate (maxCycles - 2) False) :: Signal System Bool
          downStreamReady = pure True :: Signal System Bool
          stepCount = pure 0 :: Signal System (Index 512)
          input = pure inputVec :: Signal System (Vec 64 FixedPoint)

          -- Run with DRAM weights (currentRow')
          debugInfoDRAM :: QHeadDebugInfo System
          (masterOutDRAM, qOutDRAM, validOutDRAM, readyOutDRAM, debugInfoDRAM) =
            exposeClockResetEnable
              (queryHeadProjector
                (mockDRAM arvalidSignalDRAM)
                layerIdx
                headIdx
                validIn
                downStreamReady
                stepCount
                input
                params)
              CS.systemClockGen
              CS.resetGen
              CS.enableGen

          arvalidSignalDRAM = Master.arvalid masterOutDRAM

          -- Extract all diagnostic signals for DRAM version
          dramDiagnostics :: [CycleDiagnostic]
          dramDiagnostics = flip P.map [0 .. maxCycles - 1] $ \i ->
            CycleDiagnostic
              { cycleNum   = i
              , rowIdx     = rowIdxs     P.!! i
              , state      = states      P.!! i
              , rowReset   = rowResets   P.!! i
              , rowEnable  = rowEnables  P.!! i
              , fetchValid = fetchValids P.!! i
              , firstMant  = firstMants  P.!! i
              , accumVal   = accumVals   P.!! i
              , rowResult  = rowResults  P.!! i
              , rowDone    = rowDones    P.!! i
              , qOutVec    = qOutVecs    P.!! i
              , currentRow = currentRows    P.!! i
              , currentRow'= currentRows'    P.!! i
              }
            where
              rowIdxs      = P.take maxCycles $ sample (qhRowIndex     debugInfoDRAM)
              states       = P.take maxCycles $ sample (qhState        debugInfoDRAM)
              rowResets    = P.take maxCycles $ sample (qhRowReset     debugInfoDRAM)
              rowEnables   = P.take maxCycles $ sample (qhRowEnable    debugInfoDRAM)
              fetchValids  = P.take maxCycles $ sample (qhFetchValid   debugInfoDRAM)
              firstMants   = P.take maxCycles $ sample (qhFirstMant    debugInfoDRAM)
              accumVals    = P.take maxCycles $ sample (qhAccumValue   debugInfoDRAM)
              rowResults   = P.take maxCycles $ sample (qhRowResult    debugInfoDRAM)
              rowDones     = P.take maxCycles $ sample (qhRowDone      debugInfoDRAM)
              qOutVecs     = P.take maxCycles $ sample (qhQOut         debugInfoDRAM)
              currentRows  = P.take maxCycles $ sample (qhCurrentRow     debugInfoDRAM)
              currentRows' = P.take maxCycles $ sample (qhCurrentRow'    debugInfoDRAM)
              
      it "DRAM version completes transaction" $ do
        let valids = P.take maxCycles $ sample validOutDRAM
            validIndices = DL.findIndices id valids
        P.length validIndices `shouldSatisfy` (>= 1)

      it "shows cycle-by-cycle progression (DRAM version)" $ do
        -- Print first 20 cycles for inspection
        let 
          printDiag d = do
            let
              cr = currentRow d
              cr' = currentRow' d
            P.putStrLn $ "Cycle " P.++ show (cycleNum d) P.++ ":"
            P.putStrLn $ "  rowIdx=" P.++ show (rowIdx d)
            P.putStrLn $ "  state=" P.++ show (state d)
            P.putStrLn $ "  rowReset=" P.++ show (rowReset d) P.++ 
                        ", rowEnable=" P.++ show (rowEnable d)
            P.putStrLn $ "  fetchValid=" P.++ show (fetchValid d)
            P.putStrLn $ "  firstMant=" P.++ show (firstMant d)
            P.putStrLn $ "  accumVal=" P.++ show (accumVal d)
            P.putStrLn $ "  rowResult=" P.++ show (rowResult d)
            P.putStrLn $ "  rowDone=" P.++ show (rowDone d)
            P.putStrLn $ "  qOut[0]=" P.++ show (P.head $ toList $ qOutVec d)
            --P.putStrLn $ "  currentRow=" P.++ show cr
            --P.putStrLn $ "  currentRow'=" P.++ show cr'
        
        mapM_ printDiag (P.take 20 dramDiagnostics)
        True `shouldBe` True

      it "identifies when rowReset fires" $ do
        let resetCycles = P.filter (rowReset . snd) $ P.zip [0..] dramDiagnostics
        P.putStrLn $ "\nrowReset active at cycles: " P.++ show (P.map fst resetCycles)
        True `shouldBe` True

      it "identifies when rowEnable fires" $ do
        let enableCycles = P.filter (rowEnable . snd) $ P.zip [0..] dramDiagnostics
        P.putStrLn $ "\nrowEnable active at cycles: " P.++ show (P.map fst enableCycles)
        True `shouldBe` True

      it "tracks accumulator evolution" $ do
        let accums = P.map (\d -> (cycleNum d, accumVal d)) $ P.take 20 dramDiagnostics
        P.putStrLn "\nAccumulator evolution:"
        mapM_ (\(c, a) -> P.putStrLn $ "  Cycle " P.++ show c P.++ ": " P.++ show a) accums
        True `shouldBe` True

      it "verifies row completion sequence" $ do
        let doneCycles = P.filter (rowDone . snd) $ P.zip [0..] dramDiagnostics
        P.putStrLn $ "\nrowDone asserted at cycles: " P.++ show (P.map fst doneCycles)
        P.length doneCycles `shouldBe` 8  -- Should see 8 row completions
