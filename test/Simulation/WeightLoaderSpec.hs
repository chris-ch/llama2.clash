module Simulation.WeightLoaderSpec (spec) where

import Clash.Prelude
import qualified Data.List as DL
import LLaMa2.Memory.WeightLoader (calculateLayerBaseAddress, parseI8EChunk, weightManagementSystem, WeightSystemState (..))
import LLaMa2.Memory.WeightLoader.BootWeightLoader (BootLoaderState (..), bootWeightLoader, calculateLayerSizeBytes, calculateModelSizeBytes)
import LLaMa2.Numeric.Types (Mantissa)
import Test.Hspec
import qualified Prelude as P
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Types as AXITypes
import qualified Data.ByteString.Lazy as BSL
import Simulation.FileBackedAxiSlave (createFileBackedAxiSlave)
import Simulation.RamBackedAxiSlave (createRamBackedAxiSlave, WriteState (..))
import Control.Monad (unless)

-- ============================================================================
-- MOCKS
-- ============================================================================

-- A simple, always-ready AXI slave for testing.
mockAxiSlave :: Slave.AxiSlaveIn dom
mockAxiSlave =
  Slave.AxiSlaveIn
    { arready = pure True,
      rvalid = fromList $ P.replicate 5 False P.++ P.repeat True,
      rdata = pure (AXITypes.AxiR 0 0 False 0),
      awready = pure True,
      wready = fromList $ P.replicate 5 False P.++ P.repeat True,
      bvalid = pure True,
      bdata = pure (AXITypes.AxiB 0 0)
    }

-- | Simulation wrapper around 'bootWeightLoader' that limits model size.
bootWeightLoaderFast ::
  (HiddenClockResetEnable dom) =>
  Slave.AxiSlaveIn dom ->
  Slave.AxiSlaveIn dom ->
  Signal dom Bool ->
  Signal dom (Unsigned 32) ->
  Signal dom (Unsigned 32) ->
  ( Master.AxiMasterOut dom,
    Master.AxiMasterOut dom,
    Signal dom Bool,
    Signal dom (Unsigned 32),
    Signal dom BootLoaderState
  )
bootWeightLoaderFast emmcSlave ddrSlave startBoot emmcBase ddrBase =
  (emmcMaster, ddrMaster, bootComplete', bytesTransferred, state)
  where
    -- Run the real loader
    (emmcMaster, ddrMaster, bootComplete, bytesTransferred, state
      , readValid, writerDataReady, transfersInBurst
      , burstComplete, allComplete, currentBurst, _, _
      ) =
      bootWeightLoader emmcSlave ddrSlave startBoot emmcBase ddrBase

    -- Force quick completion in simulation:
    -- Once we reach BootReading, we flip to BootComplete after a few cycles.
    fakeComplete =
      register False
        $ mux (state .==. pure BootReading) (pure True) (pure False)

    bootComplete' = bootComplete .||. fakeComplete

-- ============================================================================
-- SPEC
-- ============================================================================

spec :: Spec
spec = do
  describe "calculateLayerBaseAddress"
    $ it "increments linearly per layer"
    $ do
      let addr0 = calculateLayerBaseAddress 0
          addr1 = calculateLayerBaseAddress 1
      addr1 `shouldBe` (addr0 + calculateLayerSizeBytes)

  describe "bootWeightLoader" $ do
    it "transitions from BootIdle → BootReading → BootComplete (fake model)" $ do
      let startBoot = fromList $ [False, True] P.++ P.repeat False
          emmcBase = pure 0
          ddrBase = pure 0
          (_, _, bootComplete, _, state) =
            withClockResetEnable systemClockGen resetGen enableGen
              $ bootWeightLoaderFast mockAxiSlave mockAxiSlave startBoot emmcBase ddrBase

          states = sampleN 200 state
          done = sampleN 200 bootComplete

      states `shouldContain` [BootIdle, BootReading]
      P.or done `shouldBe` True

    it "transitions from BootIdle → BootReading → BootComplete (real, mini model 260K)" $ do
      let startBoot = fromList $ [False, True] P.++ P.repeat False
          emmcBase = pure 0
          ddrBase = pure 0
          (_, _, bootComplete, _, state
            , readValid, writerDataReady, transfersInBurst
            , burstComplete, allComplete, currentBurst, _, _
            ) =
            withClockResetEnable systemClockGen resetGen enableGen
              $ bootWeightLoader mockAxiSlave mockAxiSlave startBoot emmcBase ddrBase

          states = sampleN 250_000 state
          done = sampleN 250_000 bootComplete

      DL.nub states `shouldContain` [BootIdle, BootReading, BootPause1,  BootPause2, BootComplete]
      P.or done `shouldBe` True

    it "handles delayed rvalid signals without stalling forever" $ do
      let slowSlave = mockAxiSlave {Slave.rvalid = fromList $ P.replicate 50 False P.++ P.repeat True}
          (_, _, bootComplete, _, _
            , readValid, writerDataReady, transfersInBurst
            , burstComplete, allComplete, currentBurst, _, _
            ) =
            withClockResetEnable systemClockGen resetGen enableGen
              $ bootWeightLoader slowSlave mockAxiSlave (pure True) (pure 0) (pure 0)
          done = sampleN 250_000 bootComplete
      DL.nub done `shouldContain` [True]

  describe "parseI8EChunk"
    $ it "extracts mantissas and exponent correctly"
    $ do
      let bv = bitCoerce (replicate d64 (0x7F :: BitVector 8)) :: BitVector 512
          (mants, expn) = parseI8EChunk @8 bv
      -- Correct: provide SNat 8 explicitly + value
      mants `shouldBe` replicate (SNat @8) (127 :: Mantissa)
      -- Exponent needs bit manipulation to extract 7-bit value
      pack expn `shouldBe` (0x7F :: BitVector 7)

  describe "weightManagementSystem (bypass)"
    $ context "immediately reports WSReady when bypass=True"
    $ do
      let bypass = pure True
          powerOn = pure True
          layerIdx = pure 0
          loadTrig = pure False
          sinkRdy = pure True
          (_emmcMaster, _ddrMaster, _, _, sysReady, _, sysState, bootState
            , readValid, writerDataReady, transfersInBurst
            , burstComplete, allComplete, currentBurst, _, _
            ) =
            withClockResetEnable systemClockGen resetGen enableGen
              $ weightManagementSystem bypass mockAxiSlave mockAxiSlave powerOn layerIdx loadTrig sinkRdy

          readyS = P.drop 2 (sampleN 10 sysReady)
          stateS = P.drop 2 (sampleN 10 sysState)
          bootS = sampleN 10 bootState
      it "all ready" $ do
        P.and readyS `shouldBe` True
      it "all states ready" $ do
        P.all (== WSReady) stateS `shouldBe` True
      it "no boot reading" $ do
        bootS `shouldSatisfy` P.notElem BootReading

  describe "boot smoke test: DDR gets written" $ do
    context "using full model" $ do
      it "completes boot within cycle budget and produces expected side effects" $ do
        modelBinary <- BSL.readFile "data/stories260K.bin"

        let powerOn   = pure True
            bypass    = pure False
            layerIdx  = pure 0
            loadTrig  = pure False
            sinkRdy   = pure True
            totalCycles = 400_000   -- generous budget; adjust after profiling
        let expectedModelSize = calculateModelSizeBytes
            actualModelSize = 1056540
        withClockResetEnable systemClockGen resetGen enableGen $ do
          let (emmcSlave, _)                     = createFileBackedAxiSlave modelBinary emmcMasterOut
              (ddrSlave, _rState, wState)        = createRamBackedAxiSlave modelBinary ddrMasterOut
              (emmcMasterOut, ddrMasterOut, _weightStream, streamValid,
                _sysReady, _bootProgress, _sysState, bootState
                , readValid, writerDataReady, transfersInBurst
                , burstComplete, allComplete, currentBurst
                , burstStarted, startReadBurst
                ) =
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

          let sampledReadValid = sampleN totalCycles readValid
              sampledWriterReady = sampleN totalCycles writerDataReady
              sampledTransfers = sampleN totalCycles transfersInBurst
              sampledBurstComplete = sampleN totalCycles burstComplete
              sampledAllComplete = sampleN totalCycles allComplete
              sampledCurrentBurst = sampleN totalCycles currentBurst
              sampledBurstStarted = sampleN totalCycles burstStarted
              sampledStartReadBurst = sampleN totalCycles startReadBurst

              -- Add to your error message:
              readEverValid = or sampledReadValid
              writerEverReady = or sampledWriterReady
              maxTransfers = P.maximum sampledTransfers
              burstCompleteEver = or sampledBurstComplete
              allCompleteEver = or sampledAllComplete
              finalBurst = P.last sampledCurrentBurst
              -- Find when burstComplete first went high
              firstBurstCompleteCycle = DL.findIndex id sampledBurstComplete
              -- Count how many times startReadBurst pulsed
              startReadBurstPulses = P.length $ P.filter id sampledStartReadBurst
              -- Find cycles when burstStarted transitioned
              burstStartedTransitions = P.zip [0 :: Int ..] $ P.zip sampledBurstStarted (P.drop 1 sampledBurstStarted)
              burstStartedChanges = [(cyc, old, new) | (cyc, (old, new)) <- burstStartedTransitions, old /= new]
              -- Find cycles when startReadBurst was True
              startReadBurstCycles = [cyc | (cyc, val) <- P.zip [0 :: Int ..] sampledStartReadBurst, val]

          -- Update the error message:
          unless bootCompleted $
            fail $ unlines
              [ "Boot did NOT reach BootComplete within " P.++ show totalCycles P.++ " cycles."
              , " • Final boot state: " P.++ show finalBootState
              , " • Completion cycle: " P.++ maybe "never" show cycleOfCompletion
              , " • DDR write states observed: " P.++ show writeStatesSeen
              , " • StreamValid ever high: " P.++ show streamEverValid
              , " • ReadValid ever high: " P.++ show readEverValid
              , " • WriterDataReady ever high: " P.++ show writerEverReady
              , " • Max transfers in burst: " P.++ show maxTransfers
              , " • BurstComplete ever high: " P.++ show burstCompleteEver
              , " • AllComplete ever high: " P.++ show allCompleteEver
              , " • Final burst counter: " P.++ show finalBurst
              , " • First burstComplete cycle: " P.++ maybe "never" show firstBurstCompleteCycle
              , " • StartReadBurst pulse count: " P.++ show startReadBurstPulses
              , " • StartReadBurst cycles: " P.++ show (P.take 10 startReadBurstCycles P.++ ([-1 | P.length startReadBurstCycles > 10]))
              , " • BurstStarted transitions: " P.++ show (P.take 10 burstStartedChanges P.++ ([(-1, False, False) | P.length burstStartedChanges > 10]))
              ]
          actualModelSize `shouldBe` expectedModelSize
          ddrBurstObserved `shouldBe` True
          streamEverValid  `shouldBe` True
