module Simulation.DRAMBackedAxiSlaveSpec (spec) where

import Clash.Prelude
import Test.Hspec
import qualified Prelude as P
import Simulation.DRAMBackedAxiSlave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified  LLaMa2.Memory.AXI.Types as AXITypes
import System.Random (mkStdGen, randoms)

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
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) initMem masterOut

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
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) initMem masterOut

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
          -- AWVALID in cycle 1, WVALID in cycle 2 (1-cycle later)
          awvalid = fromList $ [False, True] P.++ P.repeat False
          awdata  = fromList $
                      [ defaultAW
                      , AXITypes.AxiAW 0 0 0 1 0   -- addr=0, len=0 (single-beat), size=3, burst=INCR, id=0
                      ] P.++ P.repeat defaultAW

          wvalid = fromList $ [False, False, True] P.++ P.repeat False
          wdata  = fromList $
                      [ defaultW
                      , defaultW
                      , AXITypes.AxiW 0x123456789ABCDEF0
                            (maxBound :: BitVector 64)
                            True
                      ] P.++ P.repeat defaultW

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
                    createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) initMem masterOut

        sampledBValid :: [Bool]
        sampledBValid = sampleN 12 (Slave.bvalid slaveIn)

      P.putStrLn $ "bValid (single write): " P.++ show sampledBValid

      P.length (filter id sampledBValid) `shouldBe` 1

      -- Extract bresp from bdata
      let
        brespSamples :: [Unsigned 2]
        brespSamples = sampleN 12 (AXITypes.bresp <$> Slave.bdata slaveIn)

      brespSamples P.!! 5 `shouldBe` 0

    it "applies configured read latency correctly" $ do
      -- readLatency = 5 → expect rvalid delayed by 6 cycles total (1 RAM + 5 extra)
      let initMem :: Vec 65536 WordData
          initMem = repeat 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
          arvalid = fromList $ [False, True] P.++ P.repeat False
          ardata  = fromList $ AXITypes.AxiAR 0 0 0 0 0 : P.repeat (AXITypes.AxiAR 0 0 0 0 0)
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
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 5 1 1) initMem masterOut

      let sampledRValid = sampleN 15 (Slave.rvalid slaveIn)
      P.putStrLn $ "rValid (latency=5): " P.++ show sampledRValid

      -- Should appear only once, around cycle 6
      let validIdxs = [i | (i,v) <- P.zip [0 :: Int ..] sampledRValid, v]
      validIdxs `shouldSatisfy` (\ix -> not (null ix) && P.head ix >= 5 && P.head ix <= 7)

    it "handles a 4-beat burst write correctly and issues a single response" $ do
      let initMem :: Vec 65536 WordData
          initMem = repeat 0
          awvalid = fromList $ [False, True] P.++ P.repeat False
          awdata  = fromList $
                      [ AXITypes.AxiAW 0 3 0 1 0  -- len=3 → 4 beats
                      , AXITypes.AxiAW 0 3 0 1 0
                      ] P.++ P.repeat (AXITypes.AxiAW 0 3 0 1 0)

          wvalid = fromList $ [False, True, True, True, True] P.++ P.repeat False
          wdata  = fromList $
                      [ AXITypes.AxiW 0xA 0xFFFFFFFFFFFFFFFF True
                      , AXITypes.AxiW 0xB 0xFFFFFFFFFFFFFFFF True
                      , AXITypes.AxiW 0xC 0xFFFFFFFFFFFFFFFF True
                      , AXITypes.AxiW 0xD 0xFFFFFFFFFFFFFFFF True
                      , AXITypes.AxiW 0xE 0xFFFFFFFFFFFFFFFF True
                      ] P.++ P.repeat (AXITypes.AxiW 0 0 False)

          masterOut = Master.AxiMasterOut
            { Master.arvalid = pure False
            , Master.ardata  = pure (AXITypes.AxiAR 0 0 0 0 0)
            , Master.rready  = pure False
            , Master.awvalid = awvalid
            , Master.awdata  = awdata
            , Master.wvalid  = wvalid
            , Master.wdata   = wdata
            , Master.bready  = pure True
            }

      let slaveIn = withClockResetEnable systemClockGen resetGen enableGen $
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) initMem masterOut

      let bValidSamples = sampleN 20 (Slave.bvalid slaveIn)
      P.putStrLn $ "bValid (burst write 4): " P.++ show bValidSamples

      -- Expect exactly one valid response after all beats complete
      P.length (filter id bValidSamples) `shouldBe` 1

    it "ignores WVALID beats that arrive before any AWVALID" $ do
      let initMem :: Vec 65536 WordData
          initMem = repeat 0
          awvalid = pure False
          awdata  = pure (AXITypes.AxiAW 0 0 0 1 0)
          wvalid  = fromList $ [True, True, False] P.++ P.repeat False
          wdata   = fromList $ AXITypes.AxiW 0x1234 0xFFFFFFFFFFFFFFFF True : P.repeat (AXITypes.AxiW 0 0 False)
          masterOut = Master.AxiMasterOut
            { Master.arvalid = pure False
            , Master.ardata  = pure (AXITypes.AxiAR 0 0 0 0 0)
            , Master.rready  = pure False
            , Master.awvalid = awvalid
            , Master.awdata  = awdata
            , Master.wvalid  = wvalid
            , Master.wdata   = wdata
            , Master.bready  = pure True
            }

      let slaveIn = withClockResetEnable systemClockGen resetGen enableGen $
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) initMem masterOut

      let wreadySamples = sampleN 10 (Slave.wready slaveIn)
      let bvalidSamples = sampleN 10 (Slave.bvalid slaveIn)
      P.putStrLn $ "wready (no AW): " P.++ show wreadySamples
      P.putStrLn $ "bvalid (no AW): " P.++ show bvalidSamples

      -- Should not accept writes (wready should remain False)
      P.or wreadySamples `shouldBe` False
      -- Should not produce a response
      P.or bvalidSamples `shouldBe` False

    it "passes randomized AXI fuzz test (100 cycles, deterministic seed)" $ do
      let gen = mkStdGen 42
          randBools = randoms gen :: [Bool]
          inf xs     = xs P.++ P.repeat False

          arvalid = fromList $ inf (takeN 100 randBools)
          awvalid = fromList $ inf (takeN 100 (P.drop 100 randBools))
          wvalid  = fromList $ inf (takeN 100 (P.drop 200 randBools))
          rready  = fromList $ inf (takeN 100 (P.drop 300 randBools))
          bready  = fromList $ inf (takeN 100 (P.drop 400 randBools))

          defaultAR = AXITypes.AxiAR 0 0 0 0 0
          defaultAW = AXITypes.AxiAW 0 0 0 1 0
          -- Write one well-defined 512-bit word: low 32 bits 0xF00DBABE, rest zero
          defaultW  = AXITypes.AxiW  (0xF00DBABE :: WordData)  0xFFFFFFFFFFFFFFFF  True

          ardata = fromList (P.replicate 100 defaultAR P.++ P.repeat defaultAR)
          awdata = fromList (P.replicate 100 defaultAW P.++ P.repeat defaultAW)
          wdata  = fromList (P.replicate 100 defaultW  P.++ P.repeat defaultW)

          initMem :: Vec 65536 WordData
          initMem = repeat 0x5555555555555555555555555555555555555555555555555555555555555555

          masterOut = Master.AxiMasterOut
            { Master.arvalid = arvalid
            , Master.ardata  = ardata
            , Master.rready  = rready
            , Master.awvalid = awvalid
            , Master.awdata  = awdata
            , Master.wvalid  = wvalid
            , Master.wdata   = wdata
            , Master.bready  = bready
            }

          slaveIn = withClockResetEnable systemClockGen resetGen enableGen $
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 2 1 1) initMem masterOut

          rs = sampleN 120 (bundle ( Slave.rvalid slaveIn
                                   , AXITypes.rdata <$> Slave.rdata slaveIn))
          bs = sampleN 120 (Slave.bvalid slaveIn)

          rBeats = [d | (True,d) <- rs]

          -- Helper: create an infinite boolean stream (first 100 pseudo-random, then False)
          takeN = P.take          -- gate on rvalid

      P.putStrLn $ "Random fuzz: rvalid first 30 = " P.++ show (P.take 30 (P.map fst rs))
      P.putStrLn $ "Random fuzz: bvalid first 30 = " P.++ show (P.take 30 bs)

      -- 1) Coverage
      P.length rBeats `shouldSatisfy` (> 0)
      P.length (filter id bs) `shouldSatisfy` (> 0)

      -- 2) rdata is either initial pattern or the written pattern
      let initPat  = 0x5555555555555555555555555555555555555555555555555555555555555555 :: WordData
          writePat = 0xF00DBABE :: WordData
      P.all (\x -> x == initPat || x == writePat) (P.take 50 rBeats) `shouldBe` True

      -- 3) No X explosions
      rBeats `shouldSatisfy` P.all (\x -> x == x)

    it "handles two sequential reads to the same address without state pollution" $ do
      let initMem :: Vec 65536 WordData
          initMem = replace (0 :: Int) 0xDEADBEEFCAFEBABEDEADBEEFCAFEBABEDEADBEEFCAFEBABEDEADBEEFCAFEBABE $
                    repeat 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

          -- First read at cycle 1, second read at cycle 10 (after first completes)
          arvalid = fromList $ [False, True] P.++ P.replicate 8 False P.++
                              [True] P.++ P.repeat False

          ardata  = fromList $ [AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 0 0 6 0 0,  -- addr=0, len=0, size=6 (64 bytes)
                              AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 0 0 6 0 0  -- Second read: same address
                              ] P.++ P.repeat (AXITypes.AxiAR 0 0 0 0 0)

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
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) initMem masterOut

      let sampledR = sampleN 20 (bundle (Slave.rvalid slaveIn,
                                        AXITypes.rdata <$> Slave.rdata slaveIn))

      let validReads = [d | (True, d) <- sampledR]

      -- Should get exactly 2 reads
      P.length validReads `shouldBe` 2

      let firstRead = P.head validReads
          secondRead = validReads P.!! 1

      -- Both reads should return the same data (no state pollution)
      firstRead `shouldBe` secondRead

      -- Both should be the test pattern we wrote to address 0
      firstRead `shouldBe` 0xDEADBEEFCAFEBABEDEADBEEFCAFEBABEDEADBEEFCAFEBABEDEADBEEFCAFEBABE

    it "handles sequential reads to different addresses correctly" $ do
      let initMem :: Vec 65536 WordData
          initMem = replace (0 :: Int) 0x1111111111111111111111111111111111111111111111111111111111111111 $
                    replace (1 :: Int) 0x2222222222222222222222222222222222222222222222222222222222222222 $
                    repeat 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

          -- Read addr 0, then addr 64 (next 64-byte block)
          arvalid = fromList $ [False, True, False, False, False, False,
                              True] P.++ P.repeat False

          ardata  = fromList $ [AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 0 0 6 0 0,   -- addr=0
                              AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 0 0 0 0 0, AXITypes.AxiAR 0 0 0 0 0,
                              AXITypes.AxiAR 64 0 6 0 0   -- addr=64
                              ] P.++ P.repeat (AXITypes.AxiAR 0 0 0 0 0)

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
                      createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 1 1) initMem masterOut

      let sampledR = sampleN 15 (bundle (Slave.rvalid slaveIn,
                                        AXITypes.rdata <$> Slave.rdata slaveIn))

      let validReads = [d | (True, d) <- sampledR]

      -- Should get exactly 2 reads
      P.length validReads `shouldBe` 2

      let firstRead = P.head validReads
          secondRead = validReads P.!! 1

      -- Reads should be different
      firstRead `shouldBe` 0x1111111111111111111111111111111111111111111111111111111111111111
      secondRead `shouldBe` 0x2222222222222222222222222222222222222222222222222222222222222222
