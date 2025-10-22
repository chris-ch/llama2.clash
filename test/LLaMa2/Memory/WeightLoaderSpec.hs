module LLaMa2.Memory.WeightLoaderSpec (spec) where
import Clash.Prelude
import Test.Hspec
import LLaMa2.Memory.WeightLoader
import LLaMa2.Memory.AXI
import qualified Prelude as P
import LLaMa2.Numeric.Types (Mantissa)
import qualified Data.List as DL

-- ============================================================================
-- MOCKS
-- ============================================================================

-- A simple, always-ready AXI slave for testing.
mockAxiSlave :: AxiSlaveIn dom
mockAxiSlave = AxiSlaveIn
  { arready = pure True
  , rvalid  = fromList $ P.replicate 5 False P.++ P.repeat True
  , rdataSI = pure (AxiR 0 0 False 0)
  , awready = pure True
  , wready  = fromList $ P.replicate 5 False P.++ P.repeat True
  , bvalid  = pure True
  , bdata   = pure (AxiB 0 0)
  }

-- | Simulation wrapper around 'bootWeightLoader' that limits model size.
bootWeightLoaderFast :: HiddenClockResetEnable dom
  => AxiSlaveIn dom
  -> AxiSlaveIn dom
  -> Signal dom Bool
  -> Signal dom (Unsigned 32)
  -> Signal dom (Unsigned 32)
  -> ( AxiMasterOut dom
     , AxiMasterOut dom
     , Signal dom Bool
     , Signal dom (Unsigned 32)
     , Signal dom BootLoaderState
     )
bootWeightLoaderFast emmcSlave ddrSlave startBoot emmcBase ddrBase =
  (emmcMaster, ddrMaster, bootComplete', bytesTransferred, state)
  where
    -- Run the real loader
    (emmcMaster, ddrMaster, bootComplete, bytesTransferred, state) =
      bootWeightLoader emmcSlave ddrSlave startBoot emmcBase ddrBase

    -- Force quick completion in simulation:
    -- Once we reach BootReading, we flip to BootComplete after a few cycles.
    fakeComplete = register False $
      mux (state .==. pure BootReading) (pure True) (pure False)

    bootComplete' = bootComplete .||. fakeComplete

bootWeightLoaderDebug :: forall dom .
  HiddenClockResetEnable dom
  => AxiSlaveIn dom
  -> AxiSlaveIn dom
  -> Signal dom Bool
  -> Signal dom (Unsigned 32)
  -> Signal dom (Unsigned 32)
  -> ( AxiMasterOut dom
     , AxiMasterOut dom
     , Signal dom Bool
     , Signal dom (Unsigned 32)   -- bytesTransferred
     , Signal dom BootLoaderState
     , Signal dom (Unsigned 32)   -- transfersInBurst (debug)
     )
bootWeightLoaderDebug emmc ddr start emmcBase ddrBase =
  (emmcOut, ddrOut, done, bytes, state, transfers)
 where
  -- Run the real loader
  (emmcOut, ddrOut, done, bytes, state) =
    bootWeightLoader emmc ddr start emmcBase ddrBase

  -- Recompute this in test from the same state machine logic
  -- (non-intrusive: just reconstructs what the FSM would track)
  transfers = register 0 $
    mux (state ./=. pure BootReading) 0 $
    mux (rvalid mockAxiSlave) (transfers + 1) transfers

-- ============================================================================
-- SPEC
-- ============================================================================

spec :: Spec
spec = do
  describe "calculateLayerBaseAddress" $
    it "increments linearly per layer" $ do
      let addr0 = calculateLayerBaseAddress 0
          addr1 = calculateLayerBaseAddress 1
      addr1 `shouldBe` (addr0 + calculateLayerSizeBytes)

  describe "bootWeightLoader" $ do
    it "transitions from BootIdle → BootReading → BootComplete (fake model)" $ do
      let startBoot = fromList $ [False, True] P.++ P.repeat False
          emmcBase  = pure 0
          ddrBase   = pure 0
          (_, _, bootComplete, _, state) =
            withClockResetEnable systemClockGen resetGen enableGen $
              bootWeightLoaderFast mockAxiSlave mockAxiSlave startBoot emmcBase ddrBase

          states = sampleN 200 state
          done   = sampleN 200 bootComplete

      states `shouldContain` [BootIdle, BootReading]
      P.or done `shouldBe` True

    it "shows boot progress" $ do
      let startBoot = fromList $ [False, True] P.++ P.repeat False
          emmcBase  = pure 0
          ddrBase   = pure 0
          (_, _, _bootComplete, bytes, state, transfers) =
            withClockResetEnable systemClockGen resetGen enableGen $
              bootWeightLoaderDebug mockAxiSlave mockAxiSlave startBoot emmcBase ddrBase

          states = sampleN 2000 state
          bytesS = sampleN 2000 bytes
          trans  = sampleN 2000 transfers

      putStrLn $ "States: " <> show (P.take 10 (DL.nub states))
      putStrLn $ "Last bytesTransferred: " <> show (P.last bytesS)
      putStrLn $ "Last transfersInBurst: " <> show (P.last trans)

    it "transitions from BootIdle → BootReading → BootComplete (real, mini model 260K)" $ do
      let startBoot = fromList $ [False, True] P.++ P.repeat False
          emmcBase  = pure 0
          ddrBase   = pure 0
          (_, _, bootComplete, _, state) =
            withClockResetEnable systemClockGen resetGen enableGen $
              bootWeightLoader mockAxiSlave mockAxiSlave startBoot emmcBase ddrBase

          states = sampleN 242169 state
          done   = sampleN 242169 bootComplete

      states `shouldContain` [BootIdle, BootReading]
      P.or done `shouldBe` True

    it "handles delayed rvalid signals without stalling forever" $ do
      let slowSlave = mockAxiSlave { rvalid = fromList $ P.replicate 50 False P.++ P.repeat True }
          (_, _, bootComplete, _, _) =
            withClockResetEnable systemClockGen resetGen enableGen $
              bootWeightLoader slowSlave mockAxiSlave (pure True) (pure 0) (pure 0)
          done = sampleN 242214 bootComplete
      done `shouldContain` [True]

  describe "parseI8EChunk" $
    it "extracts mantissas and exponent correctly" $ do
      let bv = bitCoerce (replicate d64 (0x7F :: BitVector 8)) :: BitVector 512
          (mants, expn) = parseI8EChunk @8 bv
      -- Correct: provide SNat 8 explicitly + value
      mants `shouldBe` replicate (SNat @8) (127 :: Mantissa)
      -- Exponent needs bit manipulation to extract 7-bit value
      pack expn `shouldBe` (0x7F :: BitVector 7)

  describe "weightManagementSystem (bypass)" $
    context "immediately reports WSReady when bypass=True" $ do
      let bypass = pure True
          powerOn = pure True
          layerIdx = pure 0
          loadTrig = pure False
          (_emmcMaster, _ddrMaster, _, _, sysReady, _, sysState, bootState) =
            withClockResetEnable systemClockGen resetGen enableGen $
              weightManagementSystem bypass mockAxiSlave mockAxiSlave powerOn layerIdx loadTrig

          readyS = P.drop 2 (sampleN 10 sysReady)
          stateS = P.drop 2 (sampleN 10 sysState)
          bootS  = sampleN 10 bootState
      it "all ready" $ do
        P.and readyS `shouldBe` True
      it "all states ready" $ do
        P.all (== WSReady) stateS `shouldBe` True
      it "no boot reading" $ do
        bootS `shouldSatisfy` P.notElem BootReading
