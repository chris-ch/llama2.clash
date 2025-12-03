module LLaMa2.Memory.WeightsLayoutSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.AXI.Types
import LLaMa2.Memory.WeightsLayout (axiRowFetcher, rowParser, requestCaptureStage)
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Numeric.Quantization (RowI8E(..))
import Text.Printf (printf)

spec :: Spec
spec = do
    requestCaptureUnitTests
    axiReadFSMUnitTests
    axiRowFetcherIntegrationTests
    parseRowTests

-- ============================================================================
-- Unit Tests: requestCapture
-- ============================================================================

requestCaptureUnitTests :: Spec
requestCaptureUnitTests = describe "WeightsLayout - requestCapture - Unit Tests" $ do

    context "immediate acceptance (consumer always ready)" $ do
        it "passes through single request" $ do
            let maxCycles = 10
                newReqStream = [False, True] P.++ P.replicate (maxCycles-2) False
                newReq = fromList newReqStream

                addrStream = [0, 100] P.++ P.replicate (maxCycles-2) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                consumerReady = pure True

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCaptureStage newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail
                addrSamples = sampleN maxCycles capturedAddr

            availSamples P.!! 1 `shouldBe` True
            addrSamples P.!! 1 `shouldBe` 100
            availSamples P.!! 2 `shouldBe` False

        it "captures address combinationally" $ do
            let maxCycles = 5
                newReqStream = [False, True, False, False, False]
                newReq = fromList newReqStream

                addrStream = [0, 200, 0, 0, 0 :: Unsigned 32]
                newAddr = fromList addrStream

                consumerReady = pure True

                (_reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCaptureStage newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                addrSamples = sampleN maxCycles capturedAddr

            addrSamples P.!! 1 `shouldBe` 200

    context "delayed acceptance (consumer becomes ready later)" $ do
        it "latches request when consumer is busy" $ do
            let maxCycles = 15
                newReqStream = [False, True] P.++ P.replicate (maxCycles-2) False
                newReq = fromList newReqStream

                addrStream = [0, 150] P.++ P.replicate (maxCycles-2) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                readyStream = [False] P.++ P.replicate 9 False
                           P.++ [True] P.++ P.replicate (maxCycles-11) True
                consumerReady = fromList readyStream

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCaptureStage newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail
                addrSamples = sampleN maxCycles capturedAddr

            availSamples P.!! 1 `shouldBe` True
            availSamples P.!! 5 `shouldBe` True
            availSamples P.!! 9 `shouldBe` True
            addrSamples P.!! 5 `shouldBe` 150
            addrSamples P.!! 9 `shouldBe` 150

        it "clears after consumer accepts" $ do
            let maxCycles = 15
                newReqStream = [False, True] P.++ P.replicate (maxCycles-2) False
                newReq = fromList newReqStream

                addrStream = [0, 250] P.++ P.replicate (maxCycles-2) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                readyStream = [False] P.++ P.replicate 8 False
                           P.++ [True]
                           P.++ [False]
                           P.++ P.replicate (maxCycles-12) False
                consumerReady = fromList readyStream

                (reqAvail, _) = exposeClockResetEnable
                    (requestCaptureStage newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail

            availSamples P.!! 1 `shouldBe` True
            availSamples P.!! 10 `shouldBe` True
            availSamples P.!! 12 `shouldBe` False

    context "overlapping requests" $ do
        it "queues second request while first is processing" $ do
            let maxCycles = 25
                newReqStream = [False, True, False, False, False, True]
                            P.++ P.replicate (maxCycles-6) False
                newReq = fromList newReqStream

                addrStream = [0, 100, 0, 0, 0, 200]
                          P.++ P.replicate (maxCycles-6) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                readyStream = [False] P.++ P.replicate 10 False
                           P.++ [True]
                           P.++ [False]
                           P.++ P.replicate 6 False
                           P.++ [True]
                           P.++ P.replicate (maxCycles-21) True
                consumerReady = fromList readyStream

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCaptureStage newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail
                addrSamples = sampleN maxCycles capturedAddr

            availSamples P.!! 1 `shouldBe` True
            addrSamples P.!! 1 `shouldBe` 100
            availSamples P.!! 5 `shouldBe` True
            addrSamples P.!! 5 `shouldBe` 200
            availSamples P.!! 15 `shouldBe` True
            addrSamples P.!! 15 `shouldBe` 200

-- ============================================================================
-- Unit Tests: axiReadFSM
-- ============================================================================

axiReadFSMUnitTests :: Spec
axiReadFSMUnitTests = describe "WeightsLayout - axiReadFSM - Unit Tests" $ do

    context "basic FSM behavior" $ do
        it "asserts ready when idle (after reset period)" $ do
            let maxCycles = 5
                reqAvail = pure False
                reqAddr = pure (0 :: Unsigned 32)

                mockSlave = Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = pure False
                    , rdata = pure (AxiR 0 0 False 0)
                    , awready = pure False
                    , wready = pure False
                    , bvalid = pure False
                    , bdata = pure (AxiB 0 0)
                    }

                (_, _, _, ready) = exposeClockResetEnable
                    (axiRowFetcher mockSlave reqAvail reqAddr)
                    CS.systemClockGen CS.resetGen CS.enableGen

                readySamples = sampleN maxCycles ready

            -- After reset period (cycle 2+), should be ready
            readySamples P.!! 2 `shouldBe` True

        it "starts transaction when ready and reqAvail" $ do
            let maxCycles = 10
                reqAvailStream = [False, False, False, True] P.++ P.replicate (maxCycles-4) True
                reqAvail = fromList reqAvailStream

                reqAddr = pure (1000 :: Unsigned 32)

                delaySignal n sig = exposeClockResetEnable
                    (go n sig)
                    CS.systemClockGen CS.resetGen CS.enableGen
                  where
                    go 0 s = s
                    go m s = register False (go (m-1) s)

                mockSlave masterOut' = Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = delaySignal (2 :: Int) (Master.arvalid masterOut')
                    , rdata = pure (AxiR 0xABCD 0 True 0)
                    , awready = pure False
                    , wready = pure False
                    , bvalid = pure False
                    , bdata = pure (AxiB 0 0)
                    }

                (masterOut, _, _, ready) = exposeClockResetEnable
                    (axiRowFetcher (mockSlave masterOut) reqAvail reqAddr)
                    CS.systemClockGen CS.resetGen CS.enableGen

                arValidSamples = sampleN maxCycles (Master.arvalid masterOut)
                readySamples = sampleN maxCycles ready

            DL.or arValidSamples `shouldBe` True
            -- Should de-assert ready when processing (after accepting request)
            readySamples P.!! 5 `shouldBe` False

-- ============================================================================
-- Integration Tests: axiRowFetcher
-- ============================================================================

axiRowFetcherIntegrationTests :: Spec
axiRowFetcherIntegrationTests = describe "WeightsLayout - axiRowFetcher - Integration Tests" $ do

    context "sequential fetches to same address" $ do
        let
            maxCycles = 40
            testAddress = 33345

            delayedValid :: Signal System Bool -> Signal System Bool
            delayedValid arvalid = exposeClockResetEnable
                (register False $ register False arvalid)
                CS.systemClockGen CS.resetGen CS.enableGen

            testPattern :: BitVector 512
            testPattern = pack $ (11 :: BitVector 8)
                :> (22 :: BitVector 8)
                :> iterateI (+ 1) (33 :: BitVector 8)

            mockData :: Signal System AxiR
            mockData = pure (AxiR testPattern 0 True 0)

            mockDRAM :: Signal System Bool -> Slave.AxiSlaveIn System
            mockDRAM arvalidSignal' = Slave.AxiSlaveIn
                { arready = pure True
                , rvalid = delayedValid arvalidSignal'
                , rdata = mockData
                , awready = pure False
                , wready = pure False
                , bvalid = pure False
                , bdata = pure (AxiB 0 0)
                }

            requestStream = [False, True]
                P.++ P.replicate 18 False
                P.++ [True]
                P.++ P.replicate (maxCycles - 21) False
            request = fromList requestStream :: Signal System Bool

            address = pure testAddress :: Signal System (Unsigned 32)

            (masterOut, dataOut, validOut, _) = exposeClockResetEnable
                (axiRowFetcher (mockDRAM arvalidSignal) request address)
                CS.systemClockGen CS.resetGen CS.enableGen

            arvalidSignal = Master.arvalid masterOut

            outputs = P.take maxCycles $ sample dataOut
            valids = P.take maxCycles $ sample validOut
            arvalids = P.take maxCycles $ sample arvalidSignal

            validIndices = DL.findIndices id valids
            firstValid = if not (DL.null validIndices) then DL.head validIndices else 0
            secondValid = if P.length validIndices >= 2 then validIndices P.!! 1 else 0

            firstData = outputs P.!! firstValid
            secondData = outputs P.!! secondValid

        it "issues arvalid for first request" $ do
            let arvalidCount = P.length $ P.filter id $ P.take 10 arvalids
            arvalidCount `shouldSatisfy` (> 0)

        it "completes first fetch" $ do
            P.length validIndices `shouldSatisfy` (>= 1)

        it "completes second fetch" $ do
            P.length validIndices `shouldSatisfy` (>= 2)

        it "first fetch returns correct test pattern" $ do
            firstData `shouldBe` testPattern

        it "second fetch returns same data as first (no state pollution)" $ do
            secondData `shouldBe` firstData

        it "validOut pulses correctly after requests" $ do
            P.length validIndices `shouldSatisfy` (>= 2)

    context "eight requests with generous spacing" $ do
        let
            maxCycles = 400
            cyclesPerRequest = 40
            numRequests = 8

            -- IMPORTANT: do not issue a pulse during reset (cycle 0).
            -- Shift the first pulse to cycle 1 by prepending a False.
            requestStream =
                False
                : P.concat
                    [ [n == 0 | n <- [0..cyclesPerRequest-1]]
                    | _ <- [0..numRequests-1]
                    ]
                P.++ P.repeat False
            request = fromList requestStream :: Signal System Bool

            baseAddr = 49472
            -- Keep address aligned: prepend baseAddr once so that at cycle 1
            -- (first pulse) the address is baseAddr.
            addressStream =
                baseAddr
                : P.concat
                    [ P.replicate cyclesPerRequest (baseAddr + fromIntegral (i * 64))
                    | i <- [0..numRequests-1 :: Int]
                    ]
                P.++ P.repeat 0
            address = fromList addressStream :: Signal System (Unsigned 32)

            delayedValid :: Signal System Bool -> Signal System Bool
            delayedValid arvalid = exposeClockResetEnable
                (register False arvalid)
                CS.systemClockGen CS.resetGen CS.enableGen

            latchedAddr :: Signal System Bool -> Signal System (Unsigned 32) -> Signal System (Unsigned 32)
            latchedAddr arvalid addrSig = exposeClockResetEnable
                (regEn 0 arvalid addrSig)
                CS.systemClockGen CS.resetGen CS.enableGen

            mockDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
            mockDRAM masterOut' =
                let arvalidSig = Master.arvalid masterOut'
                    addrSig    = araddr <$> Master.ardata masterOut'
                    latched    = latchedAddr arvalidSig addrSig
                    dataOut'   = (\a -> AxiR (resize (pack a :: BitVector 32)) 0 True 0) <$> latched
                in Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid  = delayedValid arvalidSig
                    , rdata   = dataOut'
                    , awready = pure False
                    , wready  = pure False
                    , bvalid  = pure False
                    , bdata   = pure (AxiB 0 0)
                    }

            (masterOut, dataOut, validOut, _) = exposeClockResetEnable
                (axiRowFetcher (mockDRAM masterOut) request address)
                CS.systemClockGen CS.resetGen CS.enableGen

            valids        = P.take maxCycles $ sample validOut
            outputs       = P.take maxCycles $ sample dataOut
            requests      = P.take maxCycles $ sample request
            arvalids      = P.take maxCycles $ sample $ Master.arvalid masterOut

            requestCycles   = [n | n <- [0..maxCycles-1], requests P.!! n]
            validCycles     = [n | n <- [0..maxCycles-1], valids   P.!! n]
            arvalidCycles   = [n | n <- [0..maxCycles-1], arvalids P.!! n]

            validAddresses = [unpack (resize (outputs P.!! n) :: BitVector 32) :: Unsigned 32
                             | n <- [0..maxCycles-1], valids P.!! n]

        it "fires 8 request pulses" $ do
            P.putStrLn $ "\nRequest cycles: " P.++ show requestCycles
            P.length requestCycles `shouldBe` 8

        it "issues 8 AXI AR transactions" $ do
            P.putStrLn $ "AR valid cycles: " P.++ show arvalidCycles
            P.length arvalidCycles `shouldBe` 8

        it "produces 8 valid outputs" $ do
            P.putStrLn $ "Valid output cycles: " P.++ show validCycles
            P.putStrLn $ "Number of valid outputs: " P.++ show (P.length validCycles)
            P.length validCycles `shouldBe` 8

        it "outputs match requested addresses" $ do
            let expectedAddrs = [baseAddr + fromIntegral (i * 64) | i <- [0..7 :: Int]]
            P.putStrLn $ "Valid addresses: " P.++ show validAddresses
            P.putStrLn $ "Expected addresses: " P.++ show expectedAddrs
            validAddresses `shouldBe` expectedAddrs

    -- Reset-safe debug trace: first pulse after reset
    describe "DEBUG: Reset-safe trace" $ do
        it "shows all signals through cycle 0-10 (first request at cycle 1)" $ do
            let maxCycles = 15

                -- Do not pulse during reset; start at cycle 1.
                request = fromList (False : True : P.replicate (maxCycles-2) False)
                address = pure (49472 :: Unsigned 32)

                mockDRAM masterOut' = Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid  = register False (Master.arvalid masterOut')
                    , rdata   = pure (AxiR 0 0 True 0)
                    , awready = pure False
                    , wready  = pure False
                    , bvalid  = pure False
                    , bdata   = pure (AxiB 0 0)
                    }

                (masterOut, _, _, ready) = exposeClockResetEnable
                    (axiRowFetcher (mockDRAM masterOut) request address)
                    CS.systemClockGen CS.resetGen CS.enableGen

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCaptureStage request address ready)
                    CS.systemClockGen CS.resetGen CS.enableGen

                reqs     = sampleN @System maxCycles request
                readys   = sampleN maxCycles ready
                avails   = sampleN maxCycles reqAvail
                addrs    = sampleN maxCycles capturedAddr
                arvalids = sampleN maxCycles (Master.arvalid masterOut)

            P.putStrLn "\n=== RESET-SAFE TRACE (first pulse at cycle 1) ==="
            P.putStrLn "Cyc | Req | Ready | ReqAvail | Addr  | ARval"
            mapM_ (\(c, r, rd, ra, ad, arv) ->
                P.putStrLn $ printf "%3d | %5s | %5s | %8s | %5d | %5s"
                    c (show r) (show rd) (show ra) ad (show arv))
                (DL.zip6 [(0::Int)..] reqs readys avails addrs arvalids)

            -- With the first pulse at cycle 1, ARvalid should appear within a few cycles.
            DL.or (P.take 6 arvalids) `shouldBe` True

-- ============================================================================
-- Unit Tests: parseRow
-- ============================================================================

parseRowTests :: Spec
parseRowTests = describe "WeightsLayout - rowParser - Unit Tests" $ do
    context "extracts mantissas correctly" $ do
        let testWord = pack $ (11 :: BitVector 8)
                :> (20 :: BitVector 8)
                :> (-2 :: BitVector 8)
                :> iterateI (+ 1) (5 :: BitVector 8)
            RowI8E { rowMantissas = mantissas, rowExponent = _exponent} = rowParser @64 testWord

        it "extracts first mantissa correctly" $ do
            head mantissas `shouldBe` 11

        it "extracts second mantissa correctly" $ do
            mantissas !! (1 :: Int) `shouldBe` 20

        it "extracts third mantissa correctly" $ do
            mantissas !! (2 :: Int) `shouldBe` (-2)
