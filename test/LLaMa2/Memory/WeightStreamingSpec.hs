module LLaMa2.Memory.WeightStreamingSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import LLaMa2.Memory.AXI.Types
import LLaMa2.Memory.WeightStreaming (axiRowFetcher, parseRow)
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
