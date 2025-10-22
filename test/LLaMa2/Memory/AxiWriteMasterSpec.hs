module LLaMa2.Memory.AxiWriteMasterSpec (spec) where

import Test.Hspec
import Clash.Prelude
import qualified Prelude as P
import LLaMa2.Memory.AxiWriteMaster (axiWriteMaster)
import LLaMa2.Memory.AXI (AxiSlaveIn(..), AxiMasterOut (..))

spec :: Spec
spec = do
  it "axiWriteMaster asserts awvalid" $ do
    let addrIn = pure 0 :: Signal System (Unsigned 32)
        dataIn = pure 0 :: Signal System (BitVector 512)
        startIn = fromList [False, True, False, False, False, False, False, False, False, False] :: Signal System Bool
        
        fakeSlave = AxiSlaveIn 
          { arready = pure False
          , rvalid = pure False
          , rdataSI = pure undefined
          , awready = pure True      -- ✅ Always accept AW
          , wready = pure True       -- ✅ Always accept W  
          , bvalid = pure True       -- ✅ Always give B response
          , bdata = pure undefined
          }
    
    let (masterOut, _writeDone, _) = withClockResetEnable systemClockGen resetGen enableGen $
          axiWriteMaster fakeSlave addrIn dataIn startIn
    
    let awValidSamples = sampleN 10 (awvalid masterOut)
        wValidSamples = sampleN 10 (wvalid masterOut)
    
    P.length (P.filter id awValidSamples) `shouldBe` 1
    P.length (P.filter id wValidSamples) `shouldBe` 1
