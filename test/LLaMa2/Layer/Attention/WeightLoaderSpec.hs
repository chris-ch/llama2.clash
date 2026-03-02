module LLaMa2.Layer.Attention.WeightLoaderSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import LLaMa2.Layer.Attention.WeightLoader
  (qWeightLoader, kWeightLoader, WeightLoaderOutput(..))
import LLaMa2.Types.ModelConfig
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import qualified Simulation.Parameters as PARAM
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified Simulation.ParamsPlaceholder as PARAM

type TestDRAMDepth = 65536

testParams :: PARAM.DecoderParameters
testParams = PARAM.decoderConst

spec :: Spec
spec = do
  describe "qWeightLoader" $ do
    it "HC path produces valid outputs" $ do
      let maxCycles = 400
          cyclesPerRequest = 40

          -- Two idle cycles at start: cycle 0 is system reset, cycle 1 has notInReset=False.
          -- First valid request fires at cycle 2 when axiRowFetcher ready=True.
          requestGroups = [(0, False), (0, False)] :
                [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqSig     = fromList (P.map fst requestPairs P.++ P.repeat 0)
          reqValidSig = fromList (P.map snd requestPairs P.++ P.repeat False)
          readySig   = pure True

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams

          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec (pure 0)
                (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, weightsOut, dvDRAM, _ready) =
            exposeClockResetEnable
              (qWeightLoader (pure 0) (realDRAM axiDRAM) 0 0
                reqSig reqValidSig readySig (pure True) testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          validsSampled = sampleN maxCycles dvDRAM
          hcMant0s = sampleN maxCycles ((!! (0 :: Int)) . rowMantissas <$> hcRowOut weightsOut)
          validMant0s = [hcMant0s P.!! n | n <- [0..maxCycles-1], validsSampled P.!! n]

      P.length validMant0s `shouldSatisfy` (>= 8)
      P.length (P.filter (/= 0) validMant0s) `shouldSatisfy` (>= 1)

    it "HC and DRAM paths produce identical outputs" $ do
      let maxCycles = 400
          cyclesPerRequest = 50

          -- Two idle cycles at start: cycle 0 is system reset, cycle 1 has notInReset=False.
          requestGroups = [(0, False), (0, False)] :
                [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqSig      = fromList (P.map fst requestPairs P.++ P.repeat 0)
          reqValidSig = fromList (P.map snd requestPairs P.++ P.repeat False)
          readySig    = pure True

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams

          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec (pure 0)
                 (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, outDRAM, dvDRAM, _readyDRAM) =
            exposeClockResetEnable
              (qWeightLoader (pure 0) (realDRAM axiDRAM) 0 0
                reqSig reqValidSig readySig (pure True) testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          validsSampled    = sampleN maxCycles dvDRAM
          hcExpSampled     = sampleN maxCycles $ rowExponent <$> hcRowOut outDRAM
          dramExpSampled   = sampleN maxCycles $ rowExponent <$> dramRowOut outDRAM
          hcMantsSampled   = sampleN maxCycles $ toList . rowMantissas <$> hcRowOut outDRAM
          dramMantsSampled = sampleN maxCycles $ toList . rowMantissas <$> dramRowOut outDRAM

          validCycles = [n | n <- [0..maxCycles-1], validsSampled P.!! n]
          matches = [ hcExpSampled P.!! n == dramExpSampled P.!! n &&
                      hcMantsSampled P.!! n == dramMantsSampled P.!! n
                    | n <- validCycles ]

      P.length validCycles `shouldSatisfy` (>= 8)
      P.and matches `shouldBe` True

    it "DRAM image contains correct Q weights" $ do
      let params = PARAM.decoderConst
          layer0 = head (PARAM.modelLayers params)
          mha0   = PARAM.multiHeadAttention layer0
          qHead0 = PARAM.qMatrix (head (PARAM.qHeads mha0))

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams params

          checkRow idx =
            let addr     = Layout.rowAddressCalculator Layout.QMatrix 0 0 idx
                baseWord = fromIntegral (addr `shiftR` 6) :: Int
                slice'   = imap (\_ k -> dramContents !! (baseWord + k))
                               (iterateI (+1) 0 :: Vec (Layout.WordsPerRow ModelDimension) Int)
                dramRow  = Layout.multiWordRowParser @ModelDimension slice'
                hcRow    = qHead0 !! idx
            in dramRow == hcRow

      checkRow 0 `shouldBe` True
      checkRow 1 `shouldBe` True

  -- ---------------------------------------------------------------------------
  describe "kWeightLoader" $ do
    it "HC and DRAM paths produce identical outputs (layer 0, kvHead 0)" $ do
      let maxCycles = 400
          cyclesPerRequest = 50

          -- Two idle cycles at start: cycle 0 is system reset, cycle 1 has notInReset=False.
          requestGroups = [(0, False), (0, False)] :
                [(i, True) : P.replicate (cyclesPerRequest - 1) (i, False)
                | i <- [0..7::Index HeadDimension]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqSig      = fromList (P.map fst requestPairs P.++ P.repeat 0)
          reqValidSig = fromList (P.map snd requestPairs P.++ P.repeat False)
          readySig    = pure True

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams testParams

          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec (pure 0)
                 (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiDRAM, outDRAM, dvDRAM, _readyDRAM) =
            exposeClockResetEnable
              (kWeightLoader (pure 0) (realDRAM axiDRAM) 0 0
                reqSig reqValidSig readySig (pure True) testParams)
              CS.systemClockGen CS.resetGen CS.enableGen

          validsSampled    = sampleN maxCycles dvDRAM
          hcExpSampled     = sampleN maxCycles $ rowExponent <$> hcRowOut outDRAM
          dramExpSampled   = sampleN maxCycles $ rowExponent <$> dramRowOut outDRAM
          hcMantsSampled   = sampleN maxCycles $ toList . rowMantissas <$> hcRowOut outDRAM
          dramMantsSampled = sampleN maxCycles $ toList . rowMantissas <$> dramRowOut outDRAM

          validCycles = [n | n <- [0..maxCycles-1], validsSampled P.!! n]
          matches = [ hcExpSampled P.!! n == dramExpSampled P.!! n &&
                      hcMantsSampled P.!! n == dramMantsSampled P.!! n
                    | n <- validCycles ]

      P.length validCycles `shouldSatisfy` (>= 8)
      P.and matches `shouldBe` True

    it "DRAM image contains correct K weights (layer 0, kvHead 0)" $ do
      let params  = PARAM.decoderConst
          layer0  = head (PARAM.modelLayers params)
          mha0    = PARAM.multiHeadAttention layer0
          kHead0  = PARAM.kMatrix (head (PARAM.kvHeads mha0))

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams params

          checkKRow idx =
            let addr     = Layout.rowAddressCalculator Layout.KMatrix 0 0 idx
                baseWord = fromIntegral (addr `shiftR` 6) :: Int
                slice'   = imap (\_ k -> dramContents !! (baseWord + k))
                               (iterateI (+1) 0 :: Vec (Layout.WordsPerRow ModelDimension) Int)
                dramRow  = Layout.multiWordRowParser @ModelDimension slice'
                hcRow    = kHead0 !! idx
            in dramRow == hcRow

      checkKRow 0 `shouldBe` True
      checkKRow 1 `shouldBe` True
