module LLaMa2.Memory.WeightStreamingSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.AXI.Types
import LLaMa2.Memory.WeightStreaming (axiRowFetcher, parseRow, requestCapture, axiReadFSM)
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
requestCaptureUnitTests = describe "WeightStreaming - requestCapture - Unit Tests" $ do

    context "immediate acceptance (consumer always ready)" $ do
        it "passes through single request" $ do
            let maxCycles = 10
                newReqStream = [False, True] P.++ P.replicate (maxCycles-2) False
                newReq = fromList newReqStream

                -- FIX: Address must be valid when request fires at cycle 1
                addrStream = [0, 100] P.++ P.replicate (maxCycles-2) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                -- Consumer always ready
                consumerReady = pure True

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCapture newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail
                addrSamples = sampleN maxCycles capturedAddr

            -- Request should be available at cycle 1
            availSamples P.!! 1 `shouldBe` True
            -- Address should be captured at cycle 1
            addrSamples P.!! 1 `shouldBe` 100
            -- Request should not remain available (consumer accepted immediately)
            availSamples P.!! 2 `shouldBe` False

        it "captures address combinationally" $ do
            let maxCycles = 5
                newReqStream = [False, True, False, False, False]
                newReq = fromList newReqStream

                addrStream = [0, 200, 0, 0, 0 :: Unsigned 32]
                newAddr = fromList addrStream

                consumerReady = pure True

                (_reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCapture newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                addrSamples = sampleN maxCycles capturedAddr

            -- Address must be available same cycle as request
            addrSamples P.!! 1 `shouldBe` 200

    context "delayed acceptance (consumer becomes ready later)" $ do
        it "latches request when consumer is busy" $ do
            let maxCycles = 15
                -- Request at cycle 1
                newReqStream = [False, True] P.++ P.replicate (maxCycles-2) False
                newReq = fromList newReqStream

                addrStream = [0, 150] P.++ P.replicate (maxCycles-2) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                -- Consumer busy for 10 cycles, then ready
                readyStream = [False] P.++ P.replicate 9 False
                           P.++ [True] P.++ P.replicate (maxCycles-11) True
                consumerReady = fromList readyStream

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCapture newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail
                addrSamples = sampleN maxCycles capturedAddr

            -- Request should be latched and stay available
            availSamples P.!! 1 `shouldBe` True
            availSamples P.!! 5 `shouldBe` True
            availSamples P.!! 9 `shouldBe` True
            -- Address should remain valid
            addrSamples P.!! 5 `shouldBe` 150
            addrSamples P.!! 9 `shouldBe` 150

        it "clears after consumer accepts" $ do
            let maxCycles = 15
                newReqStream = [False, True] P.++ P.replicate (maxCycles-2) False
                newReq = fromList newReqStream

                addrStream = [0, 250] P.++ P.replicate (maxCycles-2) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                -- Busy initially, ready at cycle 10, then busy again
                readyStream = [False] P.++ P.replicate 8 False
                           P.++ [True] -- Cycle 10: ready
                           P.++ [False] -- Cycle 11: busy (consumer accepted)
                           P.++ P.replicate (maxCycles-12) False
                consumerReady = fromList readyStream

                (reqAvail, _) = exposeClockResetEnable
                    (requestCapture newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail

            -- Should be available until consumer transitions to busy
            availSamples P.!! 1 `shouldBe` True
            availSamples P.!! 10 `shouldBe` True
            -- Should clear after consumer accepts (transitions to busy)
            availSamples P.!! 12 `shouldBe` False

    context "overlapping requests" $ do
        it "queues second request while first is processing" $ do
            let maxCycles = 25
                -- Request at cycle 1 and cycle 5
                newReqStream = [False, True, False, False, False, True]
                            P.++ P.replicate (maxCycles-6) False
                newReq = fromList newReqStream

                addrStream = [0, 100, 0, 0, 0, 200]
                          P.++ P.replicate (maxCycles-6) (0 :: Unsigned 32)
                newAddr = fromList addrStream

                -- Busy initially, ready at cycle 12, busy again at 13, ready at 20
                readyStream = [False] P.++ P.replicate 10 False
                           P.++ [True] -- Cycle 12
                           P.++ [False] -- Cycle 13: first accepted
                           P.++ P.replicate 6 False
                           P.++ [True] -- Cycle 20
                           P.++ P.replicate (maxCycles-21) True
                consumerReady = fromList readyStream

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCapture newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availSamples = sampleN maxCycles reqAvail
                addrSamples = sampleN maxCycles capturedAddr

            -- First request should be available
            availSamples P.!! 1 `shouldBe` True
            addrSamples P.!! 1 `shouldBe` 100
            -- Second request arrives while first is still pending
            availSamples P.!! 5 `shouldBe` True
            addrSamples P.!! 5 `shouldBe` 200 -- Should update to new address
            -- After first is accepted, second should still be available
            availSamples P.!! 15 `shouldBe` True
            addrSamples P.!! 15 `shouldBe` 200

-- ============================================================================
-- Unit Tests: axiReadFSM (via axiRowFetcher with mocked components)
-- ============================================================================

axiReadFSMUnitTests :: Spec
axiReadFSMUnitTests = describe "WeightStreaming - axiReadFSM - Unit Tests" $ do

    context "basic FSM behavior" $ do
        it "asserts ready when idle" $ do
            let maxCycles = 5
                -- No requests
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
                    (axiReadFSM mockSlave reqAvail reqAddr)  -- Call axiReadFSM directly
                    CS.systemClockGen CS.resetGen CS.enableGen

                readySamples = sampleN maxCycles ready

            -- Should be ready at cycle 0 (after reset)
            DL.head readySamples `shouldBe` True

        it "starts transaction when ready and reqAvail" $ do
            let maxCycles = 10
                -- Request at cycle 1
                reqAvailStream = [False, True] P.++ P.replicate (maxCycles-2) True
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
                    (axiReadFSM (mockSlave masterOut) reqAvail reqAddr)  -- Call axiReadFSM directly
                    CS.systemClockGen CS.resetGen CS.enableGen

                arValidSamples = sampleN maxCycles (Master.arvalid masterOut)
                readySamples = sampleN maxCycles ready

            -- Should assert arvalid after request
            DL.or arValidSamples `shouldBe` True
            -- Should de-assert ready when processing
            readySamples P.!! 2 `shouldBe` False

-- ============================================================================
-- Integration Tests: axiRowFetcher
-- ============================================================================

axiRowFetcherIntegrationTests :: Spec
axiRowFetcherIntegrationTests = describe "WeightStreaming - axiRowFetcher - Integration Tests" $ do

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

        it "validOut pulses exactly when expected (2-3 cycles after request)" $ do
            firstValid `shouldSatisfy` (\c -> c >= 3 && c <= 5)
            secondValid `shouldSatisfy` (\c -> c >= 22 && c <= 25)

    context "eight requests with generous spacing" $ do
        let
            maxCycles = 400
            cyclesPerRequest = 40
            numRequests = 8

            requestStream = P.concat
                [ [n == 0 | n <- [0..cyclesPerRequest-1]]
                | _ <- [0..numRequests-1]
                ] P.++ P.repeat False
            request = fromList requestStream :: Signal System Bool

            baseAddr = 49472
            addressStream = P.concat
                [ P.replicate cyclesPerRequest (baseAddr + fromIntegral (i * 64))
                | i <- [0..numRequests-1 :: Int]
                ] P.++ P.repeat 0
            address = fromList addressStream :: Signal System (Unsigned 32)

            delayedValid :: Signal System Bool -> Signal System Bool
            delayedValid arvalid = exposeClockResetEnable
                (register False arvalid)
                CS.systemClockGen CS.resetGen CS.enableGen

            latchedAddr :: Signal System Bool -> Signal System (Unsigned 32) -> Signal System (Unsigned 32)
            latchedAddr arvalid addr = exposeClockResetEnable
                (regEn 0 arvalid addr)
                CS.systemClockGen CS.resetGen CS.enableGen

            mockDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
            mockDRAM masterOut' =
                let arvalidSig = Master.arvalid masterOut'
                    addrSig = araddr <$> Master.ardata masterOut'
                    latched = latchedAddr arvalidSig addrSig
                    dataOut' = (\a -> AxiR (resize (pack a :: BitVector 32)) 0 True 0) <$> latched
                in Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = delayedValid arvalidSig
                    , rdata = dataOut'
                    , awready = pure False
                    , wready = pure False
                    , bvalid = pure False
                    , bdata = pure (AxiB 0 0)
                    }

            (masterOut, dataOut, validOut, _) = exposeClockResetEnable
                (axiRowFetcher (mockDRAM masterOut) request address)
                CS.systemClockGen CS.resetGen CS.enableGen

            valids = P.take maxCycles $ sample validOut
            outputs = P.take maxCycles $ sample dataOut
            requests = P.take maxCycles $ sample request
            arvalids = P.take maxCycles $ sample $ Master.arvalid masterOut

            requestCycles = [n | n <- [0..maxCycles-1], requests P.!! n]
            validCycles = [n | n <- [0..maxCycles-1], valids P.!! n]
            arvalidCycles = [n | n <- [0..maxCycles-1], arvalids P.!! n]

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

        it "first output arrives 2-5 cycles after first request" $ do
            let firstRequest = P.head requestCycles
                firstValid = P.head validCycles
                latency = firstValid - firstRequest
            latency `shouldSatisfy` (\l -> l >= 2 && l <= 5)

-- ============================================================================
-- Unit Tests: parseRow
-- ============================================================================

parseRowTests :: Spec
parseRowTests = describe "WeightStreaming - parseRow - Unit Tests" $ do
    context "extracts mantissas correctly" $ do
        let testWord = pack $ (11 :: BitVector 8)
                :> (20 :: BitVector 8)
                :> (-2 :: BitVector 8)
                :> iterateI (+ 1) (5 :: BitVector 8)
            RowI8E { rowMantissas = mantissas, rowExponent = _exponent} = parseRow @64 testWord

        it "extracts first mantissa correctly" $ do
            head mantissas `shouldBe` 11

        it "extracts second mantissa correctly" $ do
            mantissas !! (1 :: Int) `shouldBe` 20

        it "extracts third mantissa correctly" $ do
            mantissas !! (2 :: Int) `shouldBe` (-2)

    describe "DEBUG cycle-0" $ do
        it "prints exact cycle-by-cycle state" $ do
            let maxCycles = 10
                request = fromList (True : P.replicate 9 False)
                address = pure (49472 :: Unsigned 32)

                mockDRAM masterOut' = Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = register False (Master.arvalid masterOut')
                    , rdata = pure (AxiR 0 0 True 0)
                    , awready = pure False, wready = pure False
                    , bvalid = pure False, bdata = pure (AxiB 0 0)
                    }

                (masterOut, _, _, _) = exposeClockResetEnable
                    (axiRowFetcher (mockDRAM masterOut) request address)
                    CS.systemClockGen CS.resetGen CS.enableGen

                reqs = sampleN @System maxCycles request
                arvs = sampleN maxCycles (Master.arvalid masterOut)
                addrs = sampleN maxCycles (araddr <$> Master.ardata masterOut)

            P.putStrLn "\nCyc | Req | ARval | ARaddr"
            mapM_ (\(c,r,a,addr) -> P.putStrLn $
                printf "%3d | %3s | %5s | %d" c (show r) (show a) addr)
                (DL.zip4 [(0 :: Int)..] reqs arvs addrs)

            arvs P.!! 1 `shouldBe` True
-- Add these THREE tests to WeightStreamingSpec.hs to isolate the issue

    -- TEST 1: Check if input signal is actually present
    describe "DEBUG: Input signal" $ do
        it "verifies request signal at cycle 0" $ do
            let maxCycles = 5
                request = fromList (True : P.replicate 4 False)
                samples = sampleN @System maxCycles request

            P.putStrLn "\nInput request signal:"
            mapM_ (\(c, r) -> P.putStrLn $ printf "Cycle %d: %s" c (show r))
                (P.zip [(0::Int)..] samples)

            DL.head samples `shouldBe` True


    -- TEST 2: Check requestCapture in ISOLATION
    describe "DEBUG: requestCapture isolated" $ do
        it "shows requestCapture outputs" $ do
            let maxCycles = 5

                newReq = fromList (True : P.replicate 4 False)
                newAddr = fromList (49472 : P.replicate 4 0)
                consumerReady = pure True  -- Always ready

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCapture newReq newAddr consumerReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                reqInputs = sampleN @System maxCycles newReq
                availOutputs = sampleN maxCycles reqAvail
                addrOutputs = sampleN maxCycles capturedAddr

            P.putStrLn "\nrequestCapture isolated:"
            P.putStrLn "Cyc | InReq | OutAvail | OutAddr"
            mapM_ (\(c, ir, oa, addr) ->
                P.putStrLn $ printf "%3d | %5s | %8s | %d" c (show ir) (show oa) addr)
                (DL.zip4 [(0::Int)..] reqInputs availOutputs addrOutputs)

            -- KEY: These MUST be True/49472 at cycle 0
            DL.head availOutputs `shouldBe` True
            DL.head addrOutputs `shouldBe` 49472


    -- TEST 3: Check axiReadFSM in ISOLATION (no recursive wiring)
    describe "DEBUG: axiReadFSM isolated" $ do
        it "shows axiReadFSM behavior with direct inputs" $ do
            let maxCycles = 5

                -- Direct inputs (not from requestCapture)
                reqAvail = fromList (True : P.replicate 4 False)
                reqAddr = pure (49472 :: Unsigned 32)

                mockSlave = Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = pure False
                    , rdata = pure (AxiR 0 0 False 0)
                    , awready = pure False
                    , wready = pure False
                    , bvalid = pure False
                    , bdata = pure (AxiB 0 0)
                    }

                (masterOut, _, _, ready) = exposeClockResetEnable
                    (axiReadFSM mockSlave reqAvail reqAddr)
                    CS.systemClockGen CS.resetGen CS.enableGen

                availInputs = sampleN @System maxCycles reqAvail
                readyOutputs = sampleN maxCycles ready
                arValidOutputs = sampleN maxCycles (Master.arvalid masterOut)
                arAddrOutputs = sampleN maxCycles (araddr <$> Master.ardata masterOut)

            P.putStrLn "\naxiReadFSM isolated:"
            P.putStrLn "Cyc | InAvail | OutReady | ARvalid | ARaddr"
            mapM_ (\(c, ia, or', av, aa) ->
                P.putStrLn $ printf "%3d | %7s | %8s | %7s | %d"
                    c (show ia) (show or') (show av) aa)
                (DL.zip5 [(0::Int)..] availInputs readyOutputs arValidOutputs arAddrOutputs)

            -- At cycle 0: should see ready=True, arvalid=False (still in Idle)
            DL.head readyOutputs `shouldBe` True
            -- At cycle 1: should see arvalid=True (transitioned to WaitAR)
            arValidOutputs P.!! 1 `shouldBe` True
            arAddrOutputs P.!! 1 `shouldBe` 49472

    describe "DEBUG: axiReadFSM state transition" $ do
        it "keeps reqAvail True longer" $ do
            let maxCycles = 10

                -- Keep reqAvail True for ALL cycles
                reqAvail = pure True
                reqAddr = pure (49472 :: Unsigned 32)

                mockSlave = Slave.AxiSlaveIn
                    { arready = pure True
                    , rvalid = pure False
                    , rdata = pure (AxiR 0 0 False 0)
                    , awready = pure False
                    , wready = pure False
                    , bvalid = pure False
                    , bdata = pure (AxiB 0 0)
                    }

                (masterOut, _, _, ready) = exposeClockResetEnable
                    (axiReadFSM mockSlave reqAvail reqAddr)
                    CS.systemClockGen CS.resetGen CS.enableGen

                readyOutputs = sampleN maxCycles ready
                arValidOutputs = sampleN maxCycles (Master.arvalid masterOut)
                arAddrOutputs = sampleN maxCycles (araddr <$> Master.ardata masterOut)

            P.putStrLn "\naxiReadFSM with reqAvail=pure True:"
            P.putStrLn "Cyc | OutReady | ARvalid | ARaddr"
            mapM_ (\(c, or', av, aa) ->
                P.putStrLn $ printf "%3d | %8s | %7s | %d"
                    c (show or') (show av) aa)
                (DL.zip4 [(0::Int)..] readyOutputs arValidOutputs arAddrOutputs)

            -- Should eventually see ARvalid=True when FSM transitions
            DL.or arValidOutputs `shouldBe` True

    describe "DEBUG: Full trace with hasPending" $ do
        it "traces hasPending through the cycle-0 scenario" $ do
            let maxCycles = 10

                request = fromList (True : P.replicate 9 False)
                address = fromList (49472 : P.replicate 9 0)

                -- Simulate FSM ready behavior (False for 2 cycles, then True)
                fsmReady = fromList ([False, False, True] P.++ P.replicate 7 True)

                (reqAvail, capturedAddr) = exposeClockResetEnable
                    (requestCapture request address fsmReady)
                    CS.systemClockGen CS.resetGen CS.enableGen

                reqs = sampleN @System maxCycles request
                readys = sampleN  @System maxCycles fsmReady
                avails = sampleN maxCycles reqAvail
                addrs = sampleN maxCycles capturedAddr

            P.putStrLn "\n=== requestCapture WITH FSM TIMING ==="
            P.putStrLn "Cyc | Req | FSMReady | ReqAvail | Addr"
            mapM_ (\(c, r, fr, ra, a) ->
                P.putStrLn $ printf "%3d | %3s | %8s | %8s | %d"
                    c (show r) (show fr) (show ra) a)
                (DL.zip5 [(0 :: Int)..] reqs readys avails addrs)

            -- Assertions
            DL.head avails `shouldBe` True  -- Request latched
            avails P.!! 1 `shouldBe` True  -- Still held
            avails P.!! 2 `shouldBe` True  -- Still held when FSM ready
            addrs P.!! 2 `shouldBe` 49472  -- Correct address available
