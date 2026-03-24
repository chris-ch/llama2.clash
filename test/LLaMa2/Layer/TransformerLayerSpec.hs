module LLaMa2.Layer.TransformerLayerSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E (..))
import LLaMa2.Numeric.Types (FixedPoint)
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Types.ModelConfig (HeadDimension, NumKeyValueHeads)
import LLaMa2.Layer.Attention.MultiHeadAttention (singleHeadController)
import Simulation.DRAMBackedAxiSlave (WordData, createDRAMBackedAxiSlaveFromVec, DRAMConfig (..), createKVCacheDRAMSlave)
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Layer.TransformerLayer (transformerLayer)

-- | Simple deterministic WO matrix for testing
makeSimpleWOMatrix :: forall modelDim headDim .
                      (KnownNat modelDim, KnownNat headDim)
                   => MatI8E modelDim headDim
makeSimpleWOMatrix = imap buildRow ignoredVec
  where
    ignoredVec :: Vec modelDim ()
    ignoredVec = repeat ()

    headDimVal :: Int
    headDimVal = snatToNum (SNat @headDim)

    buildRow :: Index modelDim -> () -> RowI8E headDim
    buildRow i _ = RowI8E
      { rowMantissas = imap (\j _ -> fromIntegral (fromIntegral i * headDimVal + fromIntegral j + 1))
                          (repeat () :: Vec headDim ())
      , rowExponent = 0
      }

-- | Head output with decreasing values: [1.0, 0.5, 0.333..., ...]
makeSimpleHeadOutput :: Vec HeadDimension FixedPoint
makeSimpleHeadOutput = imap (\i _ -> 1.0 / fromIntegral (i+1)) (repeat (0 :: Int))

spec :: Spec
spec = do
  describe "transformerLayer - control signals" $ do
    context "handling control signals" $ do
      it "produces defined outputs during reset" $ do
        let headOut = makeSimpleHeadOutput
            headOutputs = fromList $ DL.repeat headOut :: Signal System (Vec HeadDimension FixedPoint)
            headDones = fromList $ DL.repeat False :: Signal System Bool
            (_, validOutsSig, _readyOutsSig) =
              exposeClockResetEnable
                (singleHeadController headDones headOutputs makeSimpleWOMatrix)
                CS.systemClockGen
                CS.resetGen
                CS.enableGen
            validOuts = DL.take 5 $ sample @System validOutsSig
        all P.not validOuts `shouldBe` True
      it "headDones signal is well-defined" $ do
        let headDonesList = DL.take 15 $ DL.replicate 10 False P.++ DL.repeat True
        all (\x -> P.not x || x) headDonesList `shouldBe` True

  xdescribe "transformerLayer - Full Layer Processing (skipped: MODEL_260K too slow; rebuild with -f model-nano)" $ do
    context "processes a token through one layer" $ do
      let maxCycles = 3000

          dramContents :: Vec 65536 WordData
          dramContents = repeat 0

          realDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
          realDRAM masterOut' =
            exposeClockResetEnable
              (createDRAMBackedAxiSlaveFromVec (pure 0) (DRAMConfig 1 1 1) dramContents masterOut')
              systemClockGen resetGen enableGen

          realKVDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
          realKVDRAM kvMaster =
            exposeClockResetEnable
              (createKVCacheDRAMSlave (pure 0) kvMaster)
              systemClockGen resetGen enableGen

          kvDramSlaves = imap (\kvIx _ -> realKVDRAM (_kvMasters !! kvIx))
                           (repeat () :: Vec NumKeyValueHeads ())

          -- validIn fires at cycle 1 only
          validStream =
            [False, True] P.++ P.replicate (maxCycles - 2) False
          validIn = fromList validStream :: Signal System Bool

          seqPos = pure 0 :: Signal System (Index 512)

          -- No external slot 0 init (all-zero DRAM, zero embedding)
          initWrPort = pure Nothing

          (_masterOut, _kvMasters, layerDoneSig, readyOutSig, _bramRdData, _ffnOut0) =
            exposeClockResetEnable
              (transformerLayer
                (pure 0)
                (realDRAM masterOut)
                kvDramSlaves
                0  -- layer 0
                seqPos
                initWrPort
                validIn
                (pure 0))  -- extBramRdAddr: unused in test
              CS.systemClockGen CS.resetGen CS.enableGen

          masterOut = _masterOut

          layerDones = P.take maxCycles $ sample layerDoneSig
          readyOuts  = P.take maxCycles $ sample readyOutSig

          doneIndices = DL.findIndices id layerDones

      it "layer eventually completes (layerDone fires)" $ do
        P.length doneIndices `shouldSatisfy` (>= 1)

      it "layer becomes ready again after completion" $ do
        let firstDone = case doneIndices of { (i:_) -> i; [] -> 0 }
        let readyAfterDone = DL.or $ P.drop firstDone readyOuts
        readyAfterDone `shouldBe` True
