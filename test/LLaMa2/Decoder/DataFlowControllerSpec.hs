module LLaMa2.Decoder.DataFlowControllerSpec (spec) where

import Clash.Prelude
import qualified Data.List as DL
import Test.Hspec
import qualified Prelude as P

import LLaMa2.Decoder.DataFlowController
  ( DataFlowController(..), DataStage(..), dataFlowController )
import LLaMa2.Types.ModelConfig (NumLayers, SequenceLength)

-- | Run the controller for N cycles, returning (layer, stage, seqPos, ready, layerValidIn)
runController
  :: Int      -- ^ number of cycles to simulate
  -> [Bool]   -- ^ softReset per cycle (infinite or longer than n)
  -> [Bool]   -- ^ ffnDone per cycle
  -> [Bool]   -- ^ classifierDone per cycle
  -> [(Index NumLayers, DataStage, Index SequenceLength, Bool, Bool)]
runController n sr fdn cdn =
  P.take n $ sampleN (n + 1) $ bundle
    ( currentLayer    ctrl
    , processingStage ctrl
    , seqPosition     ctrl
    , readyPulse      ctrl
    , layerValidIn    ctrl
    )
  where
    ctrl = exposeClockResetEnable
      ( dataFlowController
          (fromList (sr  P.++ P.repeat False))
          (fromList (fdn P.++ P.repeat False))
          (fromList (cdn P.++ P.repeat False))
      )
      systemClockGen resetGen enableGen

spec :: Spec
spec = do
  describe "DataFlowController" $ do

    describe "normal operation (no reset)" $ do
      it "starts at layer 0, ProcessingLayer, seqPos 0 with layerValidIn high" $ do
        let results = runController 4 (P.repeat False) (P.repeat False) (P.repeat False)
            (l0, s0, sp0, r0, lvi0) = results P.!! 0
        l0  `shouldBe` 0
        s0  `shouldBe` ProcessingLayer
        sp0 `shouldBe` 0
        r0  `shouldBe` False
        lvi0 `shouldBe` True   -- firstCycle pulse

      it "advances to Classifier after all layers complete" $ do
        let nLayers = natToNum @NumLayers :: Int
            n   = nLayers + 6
            sr  = P.repeat False
            -- +1 extra pulse: resetGen consumes cycle-0 ffnDone without advancing
            fdn = P.replicate (nLayers + 1) True P.++ P.repeat False
            cdn = P.repeat False
            results = runController n sr fdn cdn
            stages = P.map (\(_, s, _, _, _) -> s) results
        stages `shouldSatisfy` P.elem Classifier

      it "fires readyPulse and increments seqPos on classifierDone" $ do
        let nLayers = natToNum @NumLayers :: Int
            n   = nLayers + 10
            sr  = P.repeat False
            fdn = P.replicate (nLayers + 1) True P.++ P.repeat False
            cdn = P.replicate (nLayers + 2) False P.++ P.repeat True
            results = runController n sr fdn cdn
            readies  = P.map (\(_, _, _, r, _) -> r) results
            seqPoses = P.map (\(_, _, sp, _, _) -> sp) results
        readies `shouldSatisfy` DL.or
        let seqAfterReady = P.dropWhile (== 0) seqPoses
        seqAfterReady `shouldSatisfy` P.elem 1

    describe "soft reset" $ do
      it "immediately reverts layer, stage, and seqPos to 0 when softReset is asserted" $ do
        let nLayers = natToNum @NumLayers :: Int
            advanceCycles = nLayers
            resetCycle    = advanceCycles + 1
            totalCycles   = resetCycle + 4
            sr  = P.replicate resetCycle False P.++ [True] P.++ P.repeat False
            fdn = P.replicate advanceCycles True P.++ P.repeat False
            cdn = P.repeat False
            results = runController totalCycles sr fdn cdn
            (l, s, sp, _, _) = results P.!! (resetCycle + 1)
        l  `shouldBe` 0
        s  `shouldBe` ProcessingLayer
        sp `shouldBe` 0

      it "fires layerValidIn one cycle after softReset deasserts" $ do
        let nLayers = natToNum @NumLayers :: Int
            resetCycle  = nLayers + 1
            totalCycles = resetCycle + 4
            sr  = P.replicate resetCycle False P.++ [True] P.++ P.repeat False
            fdn = P.replicate nLayers True P.++ P.repeat False
            cdn = P.repeat False
            results  = runController totalCycles sr fdn cdn
            lvis = P.map (\(_, _, _, _, lvi) -> lvi) results
            releaseIdx = resetCycle + 1
        lvis `shouldSatisfy` \xs ->
          P.length xs > releaseIdx && xs P.!! releaseIdx

      it "does not suppress readyPulse if classifierDone arrives before reset" $ do
        let nLayers  = natToNum @NumLayers :: Int
            totalCycles = nLayers + 8
            sr  = P.replicate (nLayers + 5) False P.++ [True] P.++ P.repeat False
            fdn = P.replicate (nLayers + 1) True P.++ P.repeat False
            cdn = P.replicate (nLayers + 2) False P.++ P.repeat True
            results = runController totalCycles sr fdn cdn
            readies = P.map (\(_, _, _, r, _) -> r) results
        -- A readyPulse must have fired before the reset window
        P.take (nLayers + 5) readies `shouldSatisfy` DL.or
