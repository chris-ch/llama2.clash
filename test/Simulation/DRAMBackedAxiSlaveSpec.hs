module Simulation.DRAMBackedAxiSlaveSpec (spec) where

import Clash.Prelude
import Test.Hspec
import qualified Prelude as P
import Simulation.DRAMBackedAxiSlave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified  LLaMa2.Memory.AXI.Types as AXITypes

spec :: Spec
spec = do
  describe "DRAMBackedAxiSlave basic read/write" $ do

    it "performs a single-beat read" $ do
      -- DRAM size: 2^(addrBits) * beatBytes / wordBytes
      -- With DRAMConfig 1 1 1 → 64 KiB → 65536 / 64 = 1024 beats → but we use full Vec 65536
      let initMem :: Vec 65536 WordData
          initMem = repeat 0xDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF
          arvalid = fromList $ [False, True] P.++ P.repeat False
          ardata  = fromList $ [AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 0 0 0 0] P.++ P.repeat (AXITypes.AxiAR 0 0 0 0 0)
          rready  = pure True

          masterOut = Master.AxiMasterOut
            { Master.arvalid = arvalid
            , Master.ardata  = ardata
            , Master.rready  = rready
            , Master.awvalid = pure False
            , Master.awdata  = pure (AXITypes.AxiAW 0 0 0 0 0)
            , Master.wvalid  = pure False
            , Master.wdata   = pure (AXITypes.AxiW 0 0 False)
            , Master.bready  = pure False
            }

      let slaveIn = withClockResetEnable systemClockGen resetGen enableGen $
                      createDRAMBackedAxiSlave (DRAMConfig 1 1 1) initMem masterOut

      let sampledRValid = sampleN 5 (Slave.rvalid slaveIn)
      P.putStrLn $ "rValid: " P.++ show sampledRValid

      -- Exactly 1 valid beat expected
      let validCount = P.length $ P.filter id sampledRValid
      validCount `shouldBe` 1

    it "performs a 2-beat burst read" $ do
      let initMem :: Vec 65536 WordData
          initMem = repeat 0xCAFEBABECAFEBABECAFEBABECAFEBABECAFEBABECAFEBABECAFEBABECAFEBABE

          arvalid = fromList $ [False, True] P.++ P.repeat False
          ardata  = fromList $ [AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 1 0 0 0] P.++ P.repeat (AXITypes.AxiAR 0 1 0 0 0)
          rready  = pure True

          masterOut = Master.AxiMasterOut
            { Master.arvalid = arvalid
            , Master.ardata  = ardata
            , Master.rready  = rready
            , Master.awvalid = pure False
            , Master.awdata  = pure (AXITypes.AxiAW 0 0 0 0 0)
            , Master.wvalid  = pure False
            , Master.wdata   = pure (AXITypes.AxiW 0 0 False)
            , Master.bready  = pure False
            }

      let slaveIn = withClockResetEnable systemClockGen resetGen enableGen $
                      createDRAMBackedAxiSlave (DRAMConfig 1 1 1) initMem masterOut

      let sampledRValid = sampleN 10 (Slave.rvalid slaveIn)
      P.putStrLn $ "rValid (burst 2): " P.++ show sampledRValid

      -- Exactly 2 valid beats expected
      let validCount = P.length $ P.filter id sampledRValid
      validCount `shouldBe` 2

    it "performs a single-beat write" $ do
      let initMem :: Vec 65536 WordData
          initMem = repeat 0x0

          -- Default values for each AXI type
          defaultAR = AXITypes.AxiAR 0 0 0 0 0
          defaultAW = AXITypes.AxiAW 0 0 0 0 0
          defaultW  = AXITypes.AxiW  0 0 False

          awvalid = fromList $ False : True : P.repeat False
          awdata  = fromList $
                      defaultAW
                    : AXITypes.AxiAW 0 0 3 1 0   -- addr=0, len=0, size=3 (512-bit), burst=INCR, id=0
                    : P.repeat defaultAW

          wvalid = fromList $ False : True : P.repeat False
          wdata  = fromList $
                      defaultW
                    : AXITypes.AxiW 0x123456789ABCDEF0     -- data
                           (maxBound :: BitVector 64)  -- all bytes enabled
                           True                        -- wlast
                    : P.repeat defaultW

          bready = pure True

          masterOut = Master.AxiMasterOut
            { Master.arvalid = pure False
            , Master.ardata  = pure defaultAR
            , Master.rready  = pure False
            , Master.awvalid = awvalid
            , Master.awdata  = awdata
            , Master.wvalid  = wvalid
            , Master.wdata   = wdata
            , Master.bready  = bready
            }

      let
        slaveIn = withClockResetEnable systemClockGen resetGen enableGen $
                    createDRAMBackedAxiSlave (DRAMConfig 1 1 1) initMem masterOut

        sampledBValid :: [Bool]
        sampledBValid = sampleN 12 (Slave.bvalid slaveIn)

      P.putStrLn $ "bValid (single write): " P.++ show sampledBValid

      P.length (filter id sampledBValid) `shouldBe` 1

      -- Extract bresp from bdata
      let 
        brespSamples :: [Unsigned 2]
        brespSamples = sampleN 12 (AXITypes.bresp <$> Slave.bdata slaveIn)

      brespSamples P.!! 5 `shouldBe` 0
