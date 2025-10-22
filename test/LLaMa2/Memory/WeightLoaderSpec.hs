module LLaMa2.Memory.WeightLoaderSpec (spec) where

import Test.Hspec
import Clash.Prelude
import LLaMa2.Memory.WeightLoader
import LLaMa2.Memory.AXI
import LLaMa2.Memory.AxiReadMaster (axiBurstReadMaster)
import LLaMa2.Memory.AxiWriteMaster (axiWriteMaster)

-- Mock AXI slave for testing
mockAxiSlave :: AxiSlaveIn System
mockAxiSlave = AxiSlaveIn
  { arready = pure True  -- Always ready for read address
  , rvalid  = fromList [False, False, True, True, ...]  -- Simulate delayed rvalid
  , rdataSI = pure (AxiR 0 0 False 0)  -- Dummy data
  , awready = pure True  -- Always ready for write address
  , wready  = fromList [False, False, True, True, ...]  -- Simulate delayed wready
  , bvalid  = pure True  -- Always send write response
  , bdata   = pure (AxiB 0 0)
  }

spec :: Spec
spec = do
  describe "bootWeightLoader" $ do
    it "transitions through all states and completes" $ do
      let startBoot = fromList [False, True, False, False, ...]  -- Pulse to start
          emmcBase   = pure 0
          ddrBase    = pure 0
          (emmcMaster, ddrMaster, bootComplete, bytesTransferred, state) =
            withClockResetEnable systemClockGen resetGen enableGen $
              bootWeightLoader mockAxiSlave mockAxiSlave startBoot emmcBase ddrBase

      -- Sample state and outputs
      let states = sampleN 1000 state
          bootDone = sampleN 1000 bootComplete
          bytesXfer = sampleN 1000 bytesTransferred

      -- Assertions
      states `shouldContain` [BootReading, BootWriting, BootComplete]
      any id bootDone `shouldBe` True  -- bootComplete should pulse
      last bytesXfer `shouldBe` calculateModelSizeBytes  -- All bytes transferred

    it "handles stalled handshakes" $ do
      let stalledSlave = mockAxiSlave { rvalid = fromList [False, False, False, True, ...] }
          (_, _, bootComplete, _, _) =
            withClockResetEnable systemClockGen resetGen enableGen $
              bootWeightLoader stalledSlave mockAxiSlave (pure True) (pure 0) (pure 0)

      -- Assert bootComplete still eventually pulses
      sampleN 2000 bootComplete `shouldContain` [True]
