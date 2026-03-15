module Simulation.AxiWriteMasterSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P
import LLaMa2.Memory.AxiWriteMaster (axiWriteMaster)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Types as AXITypes
import qualified LLaMa2.Memory.AXI.Master as Master

spec :: Spec
spec = do
  it "axiWriteMaster asserts exactly one awvalid and one wvalid for a 1-beat burst" $ do
    -- Inputs
    let addrIn    = pure 0 :: Signal System (Unsigned 32)
        burstLen  = pure 0 :: Signal System (Unsigned 8)   -- 0 => 1 beat
        startIn   = fromList (False : True : P.repeat False) :: Signal System Bool
        dataIn    = pure 0 :: Signal System (BitVector 512)
        dataValid = pure True

        fakeSlave = Slave.AxiSlaveIn
          { arready = pure False
          , rvalid  = pure False
          , rdata = pure (AXITypes.AxiR 0 0 False 0)
          , awready = pure True     -- always accepts AW
          , wready  = pure True     -- always accepts W
          , bvalid  = pure True     -- always responds immediately
          , bdata   = pure (AXITypes.AxiB 0 0)
          }

        (masterOut, _writeDone, _dataReady) =
          withClockResetEnable systemClockGen resetGen enableGen $
            axiWriteMaster fakeSlave addrIn burstLen startIn dataIn dataValid

        awValidSamples = sampleN 10 (Master.awvalid masterOut)
        wValidSamples  = sampleN 10 (Master.wvalid  masterOut)

    P.length (P.filter id awValidSamples) `shouldBe` 1
    P.length (P.filter id wValidSamples)  `shouldBe` 1
