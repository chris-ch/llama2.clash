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
  :: [Bool]   -- ^ softReset per cycle
  -> [Bool]   -- ^ ffnDone per cycle
  -> [Bool]   -- ^ classifierDone per cycle
  -> [(Index NumLayers, DataStage, Index SequenceLength, Bool, Bool)]
runController sr fdn cdn =
  P.take n $ sampleN (n + 1) $ bundle
    ( currentLayer    ctrl
    , processingStage ctrl
    , seqPosition     ctrl
    , readyPulse      ctrl
    , layerValidIn    ctrl
    )
  where
    n    = P.length sr
    ctrl = exposeClockResetEnable
      ( dataFlowController
          (fromList (sr  P.++ P.repeat False))
          (fromList (fdn P.++ P.repeat False))
          (fromList (cdn P.++ P.repeat False))
      )
      systemClockGen resetGen enableGen

-- | Drive ffnDone once per layer then classifierDone, no softReset
normalInputs :: Int -> ([Bool], [Bool], [Bool])
normalInputs nLayers =
  ( P.repeat False                      -- softReset: never
  , P.concat [ [True] | _ <- [1..nLayers] ] P.++ P.repeat False  -- ffnDone: one pulse per layer
  , False : False : P.repeat True       -- classifierDone: high after layers complete
  )

spec :: Spec
spec = do
  describe "DataFlowController" $ do

    describe "normal operation (no reset)" $ do
      it "starts at layer 0, ProcessingLayer, seqPos 0 with layerValidIn high" $ do
        let results = runController (P.repeat False) (P.repeat False) (P.repeat False)
            (l0, s0, sp0, r0, lvi0) = results P.!! 0
        l0  `shouldBe` 0
        s0  `shouldBe` ProcessingLayer
        sp0 `shouldBe` 0
        r0  `shouldBe` False
        lvi0 `shouldBe` True   -- firstCycle pulse

      it "advances to Classifier after all layers complete" $ do
        let nLayers = natToNum @NumLayers :: Int
            -- one ffnDone pulse per cycle for nLayers cycles, then keep False
            sr  = P.repeat False
            fdn = P.replicate nLayers True P.++ P.repeat False
            cdn = P.repeat False
            results = runController sr fdn cdn
            stages = P.map (\(_, s, _, _, _) -> s) results
        -- Classifier stage must appear within nLayers+2 cycles
        P.take (nLayers + 2) stages `shouldSatisfy` P.elem Classifier

      it "fires readyPulse and increments seqPos on classifierDone" $ do
        let nLayers = natToNum @NumLayers :: Int
            -- advance through all layers then fire classifierDone
            sr  = P.repeat False
            fdn = P.replicate nLayers True P.++ P.repeat False
            cdn = P.replicate (nLayers + 1) False P.++ P.repeat True
            results = runController sr fdn cdn
            readies = P.map (\(_, _, _, r, _) -> r) results
            seqPoses = P.map (\(_, _, sp, _, _) -> sp) results
        -- readyPulse must fire at least once in 20 cycles
        P.take 20 readies `shouldSatisfy` P.elem True
        -- seqPos must reach 1 at some point after the ready pulse
        let seqAfterReady = P.dropWhile (== 0) (P.take 20 seqPoses)
        seqAfterReady `shouldSatisfy` P.elem 1

    describe "soft reset" $ do
      it "immediately reverts layer, stage, and seqPos to 0 when softReset is asserted" $ do
        let nLayers = natToNum @NumLayers :: Int
            -- Advance part-way through layers, then assert softReset
            advanceCycles = nLayers  -- drive some ffnDone pulses first
            resetCycle    = advanceCycles + 1
            totalCycles   = resetCycle + 4
            sr  = P.replicate resetCycle False P.++ [True] P.++ P.repeat False
            fdn = P.replicate advanceCycles True P.++ P.repeat False
            cdn = P.repeat False
            results = runController sr fdn cdn
            -- Cycle right after reset is applied (register delay: state visible next cycle)
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
            results  = runController sr fdn cdn
            lvis = P.map (\(_, _, _, _, lvi) -> lvi) results
            -- layerValidIn should pulse the cycle after softReset deasserts
            releaseIdx = resetCycle + 1
        lvis `shouldSatisfy` \xs ->
          P.length xs > releaseIdx && xs P.!! releaseIdx

      it "does not suppress readyPulse if classifierDone arrives before reset" $ do
        -- Ensure softReset doesn't retroactively affect a readyPulse that already fired
        let nLayers  = natToNum @NumLayers :: Int
            sr  = P.replicate (nLayers + 3) False P.++ [True] P.++ P.repeat False
            fdn = P.replicate nLayers True P.++ P.repeat False
            cdn = P.replicate (nLayers + 1) False P.++ P.repeat True
            results = runController sr fdn cdn
            readies = P.map (\(_, _, _, r, _) -> r) results
        -- A readyPulse must have fired before the reset window
        P.take (nLayers + 3) readies `shouldSatisfy` P.elem True
