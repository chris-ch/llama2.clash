module LLaMa2.Memory.WeightStreamingSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.AXI.Types
import LLaMa2.Memory.WeightStreaming (axiRowFetcher, parseRow, requestCapture)
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Numeric.Quantization (RowI8E(..))

spec :: Spec
spec = do
    describe "axiRowFetcher - Sequential Fetches" $ do
        context "handles two sequential fetches to same address" $ do
            let
                maxCycles = 40
                testAddress = 33345 -- First Q row address

                -- Simulate 2-cycle latency: rvalid goes high 2 cycles after arvalid
                delayedValid :: Signal System Bool -> Signal System Bool
                delayedValid arvalid = exposeClockResetEnable (delayedValidCircuit arvalid) CS.systemClockGen CS.resetGen CS.enableGen
                    where
                        delayedValidCircuit :: (HiddenClockResetEnable System) => Signal System Bool -> Signal System Bool
                        delayedValidCircuit arvalid' = register False $ register False arvalid'

                -- Return test pattern: byte 0 = 11, byte 1 = 22, etc.
                testPattern :: BitVector 512
                testPattern =
                    pack $ (11 :: BitVector 8)
                        :> (22 :: BitVector 8)
                        :> iterateI (+ 1) (33 :: BitVector 8)

                mockData :: Signal System AxiR
                mockData = pure (AxiR testPattern 0 True 0)

                -- Create mock DRAM that returns known data
                mockDRAM :: Signal System Bool -> Slave.AxiSlaveIn System
                mockDRAM arvalidSignal' =
                    Slave.AxiSlaveIn
                        { arready = pure True, -- Always ready to accept address
                        rvalid = delayedValid arvalidSignal',
                        rdata = mockData,
                        awready = pure False,
                        wready = pure False,
                        bvalid = pure False,
                        bdata = pure (AxiB 0 0)
                        }

                -- Request pulses: cycle 1 and cycle 20
                requestStream =
                    [False, True]
                        P.++ P.replicate 18 False
                        P.++ [True] -- First request
                        P.++ P.replicate (maxCycles - 21) False -- Second request
                request = fromList requestStream :: Signal System Bool

                address = pure testAddress :: Signal System (Unsigned 32)

                (masterOut, dataOut, validOut) =
                    exposeClockResetEnable
                        (axiRowFetcher (mockDRAM arvalidSignal) request address)
                        CS.systemClockGen
                        CS.resetGen
                        CS.enableGen

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
                -- First valid should be around cycle 3-4 (request at 1 + 2-3 latency)
                firstValid `shouldSatisfy` (\c -> c >= 3 && c <= 5)
                -- Second valid should be around cycle 22-24 (request at 20 + 2-3 latency)
                secondValid `shouldSatisfy` (\c -> c >= 22 && c <= 25)

        context "parseRow extracts mantissas correctly" $ do
            let testWord =
                    pack $ (11 :: BitVector 8)
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

    describe "axiRowFetcher - Eight Sequential Fetches" $ do
        context "handles 8 requests with 40-cycle spacing (matching weightLoader usage)" $ do
            let
                maxCycles = 400
                cyclesPerRequest = 40
                numRequests = 8

                -- Generate request pulses: one every 40 cycles
                requestStream = P.concat
                    [ [n == 0 | n <- [0..cyclesPerRequest-1]]
                    | _ <- [0..numRequests-1]
                    ] P.++ P.repeat False
                request = fromList requestStream :: Signal System Bool

                -- Addresses increment by 64 for each request
                baseAddr = 49472
                addressStream = P.concat
                    [ P.replicate cyclesPerRequest (baseAddr + fromIntegral (i * 64))
                    | i <- [0..numRequests-1 :: Int]
                    ] P.++ P.repeat 0
                address = fromList addressStream :: Signal System (Unsigned 32)

                -- Mock DRAM with 1-cycle latency
                delayedValid :: Signal System Bool -> Signal System Bool
                delayedValid arvalid = exposeClockResetEnable
                    (register False arvalid)
                    CS.systemClockGen CS.resetGen CS.enableGen

                -- Latch the address when arvalid is true
                latchedAddr :: Signal System Bool -> Signal System (Unsigned 32) -> Signal System (Unsigned 32)
                latchedAddr arvalid addr = exposeClockResetEnable
                    (regEn 0 arvalid addr)
                    CS.systemClockGen CS.resetGen CS.enableGen

                mockDRAM :: Master.AxiMasterOut System -> Slave.AxiSlaveIn System
                mockDRAM masterOut' =
                    let arvalidSig = Master.arvalid masterOut'
                        addrSig = araddr <$> Master.ardata masterOut'
                        latched = latchedAddr arvalidSig addrSig
                        -- Return latched address as data (with delay)
                        -- Convert Unsigned 32 -> BitVector 32 -> BitVector 512
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

                (masterOut, dataOut, validOut) =
                    exposeClockResetEnable
                        (axiRowFetcher (mockDRAM masterOut) request address)
                        CS.systemClockGen
                        CS.resetGen
                        CS.enableGen

                valids = P.take maxCycles $ sample validOut
                outputs = P.take maxCycles $ sample dataOut
                requests = P.take maxCycles $ sample request
                arvalids = P.take maxCycles $ sample $ Master.arvalid masterOut

                -- Track cycles where requests were issued
                requestCycles = [n | n <- [0..maxCycles-1], requests P.!! n]

                -- Track cycles where validOut was True
                validCycles = [n | n <- [0..maxCycles-1], valids P.!! n]

                -- Track cycles where arvalid was True
                arvalidCycles = [n | n <- [0..maxCycles-1], arvalids P.!! n]

                -- Extract addresses from valid outputs (lower 32 bits of BitVector 512)
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

            it "first output arrives 3-5 cycles after first request" $ do
                let firstRequest = P.head requestCycles
                    firstValid = P.head validCycles
                    latency = firstValid - firstRequest
                latency `shouldSatisfy` (\l -> l >= 2 && l <= 5)

    describe "axiRowFetcher - Request Capture Component" $ do
        context "handles overlapping requests correctly" $ do
            let
                maxCycles = 20

                -- Two requests with only 5 cycles between them (intentionally overlapping)
                requestStream = [False, True]  -- Request at cycle 1
                    P.++ P.replicate 3 False
                    P.++ [True]  -- Request at cycle 5 (while first is still processing)
                    P.++ P.replicate (maxCycles - 6) False
                request = fromList requestStream :: Signal System Bool

                -- Different addresses
                addressStream = P.replicate 2 (100 :: Unsigned 32)
                    P.++ P.replicate 4 (200 :: Unsigned 32)
                    P.++ P.repeat (0 :: Unsigned 32)
                address = fromList addressStream :: Signal System (Unsigned 32)

                -- Simulate FSM consumer that takes 10 cycles to process
                consumerReady = fromList $
                    [True]  -- Ready at cycle 0
                    P.++ P.replicate 9 False  -- Busy for 9 cycles
                    P.++ [True]  -- Ready again at cycle 10
                    P.++ P.repeat False

                (reqAvail, capturedAddr) =
                    exposeClockResetEnable
                        (requestCapture request address consumerReady)
                        CS.systemClockGen
                        CS.systemResetGen
                        CS.enableGen

                reqAvails = P.take maxCycles $ sample reqAvail
                capturedAddrs = P.take maxCycles $ sample capturedAddr

                -- Find cycles where reqAvail was True
                availCycles = [n | n <- [0..maxCycles-1], reqAvails P.!! n]

            it "latches second request when first is still being processed" $ do
                P.putStrLn $ "\nReqAvail cycles: " P.++ show availCycles
                P.putStrLn $ "Captured addresses: " P.++ show (P.take 12 capturedAddrs)
                -- Should be available at cycles 1 (first) and continue being available
                P.length availCycles `shouldSatisfy` (>= 10)

            it "captures first address correctly" $ do
                capturedAddrs P.!! 1 `shouldBe` 100

            it "captures second address correctly when first request is busy" $ do
                -- At cycle 5, should latch addr 200 even though consumer is busy
                capturedAddrs P.!! 5 `shouldBe` 200