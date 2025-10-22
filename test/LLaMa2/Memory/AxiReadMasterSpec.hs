module LLaMa2.Memory.AxiReadMasterSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P
import LLaMa2.Memory.AxiReadMaster (axiReadMaster)
import LLaMa2.Memory.AXI (AxiSlaveIn(..), AxiMasterOut (..))

spec :: Spec
spec = do
  it "axiReadMaster asserts arvalid" $ do
    let addrIn = pure 0 :: Signal System (Unsigned 32)
        startIn = fromList [False, True, False, False, False, False, False, False, False, False] :: Signal System Bool  -- ✅ DELAY 1 CYCLE
        
        fakeSlave = AxiSlaveIn 
          { arready = pure True
          , rvalid = pure False
          , rdataSI = pure undefined
          , awready = pure False
          , wready = pure False
          , bvalid = pure False
          , bdata = pure undefined
          }
    
    let (masterOut, _, _, _) = withClockResetEnable systemClockGen resetGen enableGen $
          axiReadMaster fakeSlave addrIn startIn
    
    let arValidSamples = sampleN 10 (arvalid masterOut)  -- ✅ 10 cycles
    
    P.length (P.filter id arValidSamples) `shouldBe` 1
