module Simulation.BootThenLoadSpec (spec) where

import Clash.Prelude
import qualified Data.ByteString.Lazy as BSL

import Test.Hspec

import Simulation.FileBackedAxiSlave (createFileBackedAxiSlave)
import Simulation.RamBackedAxiSlave (createRamBackedAxiSlave, WriteState (..))
import LLaMa2.Memory.WeightLoader
    ( BootLoaderState(BootComplete), weightManagementSystem )
import Control.Monad (unless)
import qualified Prelude as P
import qualified Data.List as DL


spec :: Spec
spec = do

  describe "Boot smoke test: DDR gets written" $ do
    context "using full model" $ do
      it "completes boot within cycle budget and produces expected side effects" $ do
        modelBinary <- BSL.readFile "data/stories260K.bin"

        let powerOn   = pure True
            bypass    = pure False
            layerIdx  = pure 0
            loadTrig  = pure False
            sinkRdy   = pure True
            totalCycles = 2_000_000   -- generous budget; adjust after profiling

        withClockResetEnable systemClockGen resetGen enableGen $ do
          let (emmcSlave, _)                     = createFileBackedAxiSlave modelBinary emmcMasterOut
              (ddrSlave, _rState, wState)        = createRamBackedAxiSlave modelBinary ddrMasterOut
              (emmcMasterOut, ddrMasterOut, _weightStream, streamValid,
                _sysReady, _bootProgress, _sysState, bootState) =
                  weightManagementSystem bypass emmcSlave ddrSlave
                                        powerOn layerIdx loadTrig sinkRdy

          let sampledBoot  = sampleN totalCycles bootState
              sampledWrite = sampleN totalCycles wState
              sampledValid = sampleN totalCycles streamValid

              bootCompleted    = BootComplete `elem` sampledBoot
              ddrBurstObserved = WBursting `elem` sampledWrite
              streamEverValid  = or sampledValid

              -- helpers for a rich failure message
              finalBootState   = P.last sampledBoot
              writeStatesSeen  = DL.nub sampledWrite
              cycleOfCompletion = DL.elemIndex BootComplete sampledBoot

          unless bootCompleted $
            fail $ unlines
              [ "Boot did NOT reach BootComplete within " P.++ show totalCycles P.++ " cycles."
              , "  • Final boot state: " P.++ show finalBootState
              , "  • Completion cycle: " P.++ maybe "never" show cycleOfCompletion
              , "  • DDR write states observed: " P.++ show writeStatesSeen
              , "  • StreamValid ever high: " P.++ show streamEverValid
              ]

          ddrBurstObserved `shouldBe` True
          streamEverValid  `shouldBe` True
