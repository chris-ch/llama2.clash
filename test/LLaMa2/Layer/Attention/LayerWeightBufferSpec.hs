module LLaMa2.Layer.Attention.LayerWeightBufferSpec (spec) where

import Clash.Prelude
import LLaMa2.Memory.I8EDynamicRower (dynamicRower)
import LLaMa2.Memory.WeightLoaderAddressingExtended
  ( LayerSeg (..),
    rowsInSeg,
  )
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Types.ModelConfig (ModelDimension, HeadDimension, HiddenDimension, NumQueryHeads, NumKeyValueHeads)

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Create synthetic beat data for testing (all 0x7F for easy verification)
syntheticBeats :: Int -> Signal dom (BitVector 512)
syntheticBeats n = fromList $ P.replicate n (bitCoerce (replicate d64 0x7F :: Vec 64 (BitVector 8))) P.++ P.repeat 0

spec :: Spec
spec = do
    describe "dynamicRower alignment" $ do
        it "mdRowValid pulses for Q segments only" $ do
            let segSeq = fromList $ [SegQ, SegQ, SegWO, SegQ, SegQ] P.++ P.repeat SegWO
                beatData = fromList $ P.replicate 64 0x7F P.++ P.repeat 0
                beatValid = pure True

                -- Run the circuit under a simulated clock/reset context
                (_, mdRowValid, rowDoneExt, _) =
                    withClockResetEnable systemClockGen resetGen enableGen
                        $ dynamicRower
                        (SNat @64)
                        (SNat @8)
                        (SNat @172)
                        beatData
                        beatValid
                        segSeq

                validSamples = sampleN 10 mdRowValid
                doneSamples = sampleN 10 rowDoneExt

            P.length (P.filter id validSamples) `shouldBe` 2
            P.length (P.filter id doneSamples) `shouldBe` 8

    describe "layerAddressGenerator rowsInSeg" $ do
        it "computes correct row counts" $ do
            rowsInSeg SegQ `shouldBe` 288
            rowsInSeg SegK `shouldBe` 288
            rowsInSeg SegV `shouldBe` 288
            rowsInSeg SegW1 `shouldBe` 768
            rowsInSeg SegRmsAtt `shouldBe` 1
            
    describe "layerAddressGenerator rowsInSeg" $ do
        it "computes correct row counts from model Nats" $ do
            let qRows  = fromInteger (natToNum @NumQueryHeads * natToNum @HeadDimension) :: Unsigned 16
                kRows  = fromInteger (natToNum @NumKeyValueHeads * natToNum @HeadDimension) :: Unsigned 16
                vRows  = kRows
                w1Rows = fromInteger (natToNum @HiddenDimension) :: Unsigned 16
                rms1   = 1 :: Unsigned 16
            rowsInSeg SegQ `shouldBe` qRows
            rowsInSeg SegK `shouldBe` kRows
            rowsInSeg SegV `shouldBe` vRows
            rowsInSeg SegW1 `shouldBe` w1Rows
            rowsInSeg SegRmsAtt `shouldBe` rms1

    describe "dynamicRower (constant SegQ)" $ do
        it "emits floor(totalBytes/rowLen) md rows" $ do
            let rowLen = fromInteger (natToNum @ModelDimension + 1) :: Int
                beats  = 200  -- arbitrary
                totalBytes = 64 * beats
                expectedRows = totalBytes `div` rowLen

                beatData  = syntheticBeats beats
                beatValid = fromList $ P.replicate beats True P.++ P.repeat False
                segSeq    = pure SegQ

                (_mdRowOut, mdRowValid, _rowDoneExt, _sinkReady) =
                    withClockResetEnable systemClockGen resetGen enableGen $
                        dynamicRower (SNat @ModelDimension) (SNat @HeadDimension) (SNat @HiddenDimension)
                                    beatData beatValid segSeq

                mdValidSamples = sampleN (beats + expectedRows + 64) mdRowValid

            P.length (filter id mdValidSamples) `shouldBe` expectedRows
