module LLaMa2.Layer.Attention.LayerWeightBufferSpec (spec) where

import Clash.Prelude
import LLaMa2.Memory.I8EDynamicRower (dynamicRower)
import LLaMa2.Memory.WeightLoaderAddressingExtended
  ( LayerSeg (..),
    rowsInSeg, layerAddressGenerator, LayerAddress (seg),
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

-- Create test address sequence manually
testSegSeq :: [LayerSeg]
testSegSeq = [SegQ, SegQ, SegWO, SegQ, SegQ] P.++ P.replicate 10 SegWO

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

    describe "dynamicRower + layerAddressGenerator alignment" $ do
        context "mdRowValid pulses exactly for Q segments with correct row count" $ do
            let
                beatData   = syntheticBeats (P.length testSegSeq * 3)  -- Enough beats
                beatValid  = pure True
                resetPulse = fromList $ [True, False] P.++ P.repeat False

                -- Run address generator first (seeded)
                (addrSeed, _) = withClockResetEnable systemClockGen resetGen enableGen $
                    layerAddressGenerator (pure False) resetPulse

                segSeed = seg <$> addrSeed

                -- Run dynamic rower
                (_mdRowOut, mdRowValid, rowDoneExt, _) =
                    withClockResetEnable systemClockGen resetGen enableGen $
                    dynamicRower (SNat @ModelDimension) (SNat @HeadDimension) (SNat @HiddenDimension)
                        beatData beatValid segSeed

                -- Sample for enough cycles to cover sequence
                validSamples  = sampleN (P.length testSegSeq * 4) mdRowValid
                doneSamples   = sampleN (P.length testSegSeq * 4) rowDoneExt

                -- Count Q segments in sequence (should be 4)
                qSegmentCount = P.length $ P.filter (== SegQ) testSegSeq
                -- Expected mdRowValid pulses = rowsInSeg SegQ (but limited by sequence length)
                expectedValidPulses = min qSegmentCount (fromIntegral $ rowsInSeg SegQ)

            it "we get the right number of valid pulses" $ do
                P.length (P.filter id validSamples) `shouldBe` expectedValidPulses
            it "rowDoneExt should pulse for ALL segments" $ do
                P.length (P.filter id doneSamples) `shouldSatisfy` (>= P.length testSegSeq)
