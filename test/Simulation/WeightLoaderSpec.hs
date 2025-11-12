module Simulation.WeightLoaderSpec (spec) where

import Clash.Prelude
import LLaMa2.Memory.WeightLoader
  ( calculateLayerBaseAddress
  , parseI8EChunk
  , weightManagementSystem
  , WeightSystemState (..)
  , calculateLayerSizeBytes
  )
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import Test.Hspec
import qualified Prelude as P
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Types as AXITypes
import Simulation.DRAMBackedAxiSlave (createDRAMBackedAxiSlave, packRowToWord, buildMemoryFromParams)
import qualified Clash.Sized.Vector as Vec
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified Simulation.ParamsPlaceholder as PARAM
import qualified Simulation.Parameters as PARAM
import LLaMa2.Types.ModelConfig (ModelDimension, SequenceLength, RotaryPositionalEmbeddingDimension)
import qualified Data.List as DL
import Numeric (showHex)

-- ============================================================================
-- MOCKS
-- ============================================================================

-- A simple, always-ready AXI slave for testing.
mockAxiSlave :: Slave.AxiSlaveIn dom
mockAxiSlave =
  Slave.AxiSlaveIn
    { arready = pure True
    , rvalid = fromList $ P.replicate 5 False P.++ P.repeat True
    , rdata = pure (AXITypes.AxiR 0 0 False 0)
    , awready = pure True
    , wready = fromList $ P.replicate 5 False P.++ P.repeat True
    , bvalid = pure True
    , bdata = pure (AXITypes.AxiB 0 0)
    }

-- ============================================================================
-- SPEC
-- ============================================================================

spec :: Spec
spec = do
  describe "calculateLayerBaseAddress" $ do
    it "increments linearly per layer" $ do
      let addr0 = calculateLayerBaseAddress 0
          addr1 = calculateLayerBaseAddress 1
      addr1 `shouldBe` (addr0 + calculateLayerSizeBytes)

    it "produces different addresses for different layers" $ do
      let addr0 = calculateLayerBaseAddress 0
          addr1 = calculateLayerBaseAddress 1
          addr2 = calculateLayerBaseAddress 2
      addr0 `shouldNotBe` addr1
      addr1 `shouldNotBe` addr2
      addr0 `shouldNotBe` addr2

  describe "parseI8EChunk" $ do
    it "extracts mantissas and exponent correctly" $ do
      let bv = bitCoerce (replicate d64 (0x7F :: BitVector 8)) :: BitVector 512
          (mants, expn) = parseI8EChunk @8 bv
      -- Check mantissas
      mants `shouldBe` replicate (SNat @8) (127 :: Mantissa)
      -- Check exponent (7-bit value)
      pack expn `shouldBe` (0x7F :: BitVector 7)

    it "handles different bit patterns" $ do
      let
          vec :: Vec 64 (BitVector 8)
          vec = Vec.replicate d32 0xAA Vec.++ Vec.replicate d32 0x55

          bv = pack vec
          (mants, _expn) = parseI8EChunk @16 bv
      P.length (filter (== 170) (toList mants)) `shouldBe` 16

    it "extracts exponent from correct position" $ do
      let
          -- Index 10 as a runtime Int
          idx :: Index 64
          idx = 10

          -- 64 bytes, all 0x00 except byte #10 which is 0x3F
          vec :: Vec 64 (BitVector 8)
          vec = Vec.replace idx 0x3F (Vec.replicate d64 0x00)

          bv :: BitVector 512
          bv = pack vec

          (_mants, expn) = parseI8EChunk @10 bv

      -- Exponent (lower 7 bits of byte 10) must be 0x3F
      pack expn `shouldBe` (0x3F :: BitVector 7)

  describe "weightManagementSystem (idle behavior)" $ do
    context "when load is not triggered" $ do
      let
          layerIdx = pure 0
          startStream = pure False
          sinkRdy = pure True
          ( _ddrMaster, _, _, sysState
            ) =
              withClockResetEnable systemClockGen resetGen enableGen
                $ weightManagementSystem mockAxiSlave startStream layerIdx sinkRdy
          stateS = P.drop 2 (sampleN 10 sysState)

      it "stays in WSReady state" $ do
        P.all (== WSReady) stateS `shouldBe` True

  describe "weightManagementSystem (loading behavior)" $ do
    context "using full model from file" $ do
      it "correctly calculates layer size" $ do
        let layerSize = calculateLayerSizeBytes
        -- Layer size should be non-zero and reasonable
        layerSize `shouldSatisfy` (> 0)
        layerSize `shouldSatisfy` (< 10_000_000) -- sanity check: < 10MB per layer

      it "DIAGNOSTIC: system state responds to trigger" $ do
        let
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False :: Signal System Bool
            sinkRdy = pure True

        let (_ddrMaster, _, _, sysState) =
              withClockResetEnable systemClockGen resetGen enableGen
                $ weightManagementSystem mockAxiSlave startStream layerIdx sinkRdy

        let sampledState = sampleN 100 sysState

        -- Print for debugging
        P.putStrLn $ "First 30 states: " P.++ show (P.take 30 sampledState)

        -- Check if we ever leave WSReady
        let everStreaming = WSStreaming `P.elem` sampledState

        everStreaming `shouldBe` True

      it "DIAGNOSTIC: with real DDR, does streamer start?" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True

        let (streamValid, sysState, ddrMaster) =
              withClockResetEnable systemClockGen resetGen enableGen $
                let ddrSlave = createDRAMBackedAxiSlave params ddrMaster'
                    (ddrMaster', _, streamValid', sysState') =
                      weightManagementSystem ddrSlave startStream layerIdx sinkRdy
                in (streamValid', sysState', ddrMaster')

        let sampledValid = sampleN 1000 streamValid
            sampledState = sampleN 1000 sysState
            -- Check the AXI signals
            sampledArValid = sampleN 1000 (Master.arvalid ddrMaster)

        P.putStrLn $ "States (0-30): " P.++ show (P.take 30 sampledState)
        P.putStrLn $ "Valid (0-30): " P.++ show (P.take 30 sampledValid)
        P.putStrLn $ "AXI arvalid (0-30): " P.++ show (P.take 30 sampledArValid)

        let stateChanges = WSStreaming `P.elem` sampledState
            axiActive = P.or sampledArValid

        stateChanges `shouldBe` True
        axiActive `shouldBe` True

      it "DIAGNOSTIC: count valid beats over 50K cycles" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True

        let (streamValid, sysState) =
              withClockResetEnable systemClockGen resetGen enableGen $
                let ddrSlave = createDRAMBackedAxiSlave params ddrMaster'
                    (ddrMaster', _, streamValid', sysState') =
                      weightManagementSystem ddrSlave startStream layerIdx sinkRdy
                in (streamValid', sysState')

        let sampledValid = sampleN 50000 streamValid
            sampledState = sampleN 50000 sysState

            validCount = P.length $ P.filter id sampledValid
            streamingCount = P.length $ P.filter (== WSStreaming) sampledState

        P.putStrLn $ "Valid beats in 50K cycles: " P.++ show validCount
        P.putStrLn $ "Streaming cycles: " P.++ show streamingCount

        validCount `shouldSatisfy` (> 0)

      it "DIAGNOSTIC: check AXI handshake over time" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True

        let (streamValid, ddrMaster, ddrSlave) =
              withClockResetEnable systemClockGen resetGen enableGen $
                let ddrSlave' = createDRAMBackedAxiSlave params ddrMaster'
                    (ddrMaster', _, streamValid', _) =
                      weightManagementSystem ddrSlave' startStream layerIdx sinkRdy
                in (streamValid', ddrMaster', ddrSlave')

        let sampledValid = sampleN 100 streamValid
            sampledArValid = sampleN 100 (Master.arvalid ddrMaster)
            sampledArReady = sampleN 100 (Slave.arready ddrSlave)
            sampledRValid = sampleN 100 (Slave.rvalid ddrSlave)
            sampledRReady = sampleN 100 (Master.rready ddrMaster)

        P.putStrLn $ "streamValid: " P.++ show (P.take 30 sampledValid)
        P.putStrLn $ "arvalid:     " P.++ show (P.take 30 sampledArValid)
        P.putStrLn $ "arready:     " P.++ show (P.take 30 sampledArReady)
        P.putStrLn $ "rvalid:      " P.++ show (P.take 30 sampledRValid)
        P.putStrLn $ "rready:      " P.++ show (P.take 30 sampledRReady)

        True `shouldBe` True

      it "produces consistent stream during burst" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True
            -- INCREASED: Need more cycles to actually see streaming happen
            totalCycles = 50_000

        let (streamValid, sysState) =
              withClockResetEnable systemClockGen resetGen enableGen $
                let ddrSlave = createDRAMBackedAxiSlave params ddrMaster
                    (ddrMaster, _, streamValid', sysState') =
                      weightManagementSystem ddrSlave startStream layerIdx sinkRdy
                in (streamValid', sysState')

        let
            sampledValid = sampleN totalCycles streamValid
            sampledState = sampleN totalCycles sysState

            -- Count valid beats
            validBeats = P.length $ P.filter id sampledValid

            -- Count streaming cycles
            streamingCycles = P.length $ P.filter (== WSStreaming) sampledState

        -- Should have many valid beats during streaming
        validBeats `shouldSatisfy` (> 100)
        streamingCycles `shouldSatisfy` (> 100)

    context "with backpressure" $ do
      it "respects sinkReady signal" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 0
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            -- Sink ready alternates: ready for 2 cycles, not ready for 2 cycles
            sinkRdy = fromList $ P.cycle [True, True, False, False] :: Signal System Bool
            totalCycles = 100_000 -- need more cycles due to backpressure

        let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
              let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                  (ddrMasterOut, _weightStream, streamValid, _sysState) =
                    weightManagementSystem ddrSlave startStream layerIdx sinkRdy
              (streamValid, sinkRdy)

        let (streamValid, _sinkRdy) = simulation

            sampledValid :: [Bool]
            sampledValid = sampleN totalCycles streamValid

            sampledSinkRdy :: [Bool]
            sampledSinkRdy = sampleN totalCycles sinkRdy

            -- Stream should only be valid when sink is ready (or shortly after)
            streamEverValid = P.or sampledValid

            -- Count how many times valid occurs when sink not ready
            -- (should be minimal - only pipeline delays)
            invalidWhileNotReady = P.length $
              P.filter (\(v, sr) -> v && not sr) $
              P.zip sampledValid sampledSinkRdy

        streamEverValid `shouldBe` True
        -- Allow some pipeline delay, but should be < 10% of total valid beats
        let totalValidBeats = P.length $ P.filter id sampledValid
        invalidWhileNotReady `shouldSatisfy` (< (totalValidBeats `P.div` 10))

    it "completes successfully with backpressure" $ do
      let
          params = PARAM.decoderConst
          layerIdx = pure 0
          startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
          sinkRdy = fromList $ P.cycle [True, False]
          totalCycles = 100_000

      let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
            let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                (ddrMasterOut, _, streamValid, sysState) =
                  weightManagementSystem ddrSlave startStream layerIdx sinkRdy
            (streamValid, sysState)

      let (streamValid, sysState) = simulation
          sampledValid = sampleN totalCycles streamValid
          sampledState = sampleN totalCycles sysState

          validCount = P.length $ P.filter id sampledValid
          streamingCount = P.length $ P.filter (== WSStreaming) sampledState
          completesEventually = WSReady `P.elem` P.drop 10000 sampledState

      validCount `shouldSatisfy` (> 100)
      streamingCount `shouldSatisfy` (> 100)
      completesEventually `shouldBe` True

    context "loading different layers" $ do
      it "loads layer 1 successfully" $ do
        let
            params = PARAM.decoderConst
            layerIdx = pure 1
            startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
            sinkRdy = pure True
            totalCycles = 50_000

        let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
              let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                  (ddrMasterOut, _weightStream, streamValid, sysState) =
                    weightManagementSystem ddrSlave startStream layerIdx sinkRdy
              (streamValid, sysState)

        let (streamValid, sysState) = simulation
            sampledValid = sampleN totalCycles streamValid
            sampledState = sampleN totalCycles sysState

            streamEverValid = P.or sampledValid
            systemWentStreaming = WSStreaming `P.elem` sampledState

        streamEverValid `shouldBe` True
        systemWentStreaming `shouldBe` True

  describe "state transitions" $ do
    it "transitions from Ready -> Streaming -> Ready" $ do
      let
          params = PARAM.decoderConst
          layerIdx = pure 0
          startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
          sinkRdy = pure True
          totalCycles = 50_000

      let simulation = withClockResetEnable systemClockGen resetGen enableGen $ do
            let ddrSlave = createDRAMBackedAxiSlave params ddrMasterOut
                (ddrMasterOut, _weightStream, _streamValid, sysState) =
                  weightManagementSystem ddrSlave startStream layerIdx sinkRdy
            sysState

      let sysState = simulation
          sampledState = sampleN totalCycles sysState
          states = P.dropWhile (== WSReady) sampledState

          -- After initial ready, should enter streaming
          firstTransition = if P.null states then WSReady else P.head states

          -- Eventually should return to ready
          eventuallyReady = WSReady `P.elem` P.drop 100 states

      firstTransition `shouldBe` WSStreaming
      eventuallyReady `shouldBe` True

    it "does not start streaming without trigger" $ do
      let
          layerIdx = pure 0
          startStream = pure False -- Never trigger
          sinkRdy = pure True
          totalCycles = 1000

      let (_ddrMaster, _, _, sysState) =
            withClockResetEnable systemClockGen resetGen enableGen
              $ weightManagementSystem mockAxiSlave startStream layerIdx sinkRdy

      let sampledState = sampleN totalCycles sysState
          allReady = P.all (== WSReady) sampledState

      allReady `shouldBe` True

    it "DIAGNOSTIC: verify non-zero weight data is streamed" $ do
      let
          params = PARAM.decoderConst
          layerIdx = pure 0
          startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
          sinkRdy = pure True
          totalCycles = 50_000

      let (weightStream, streamValid) =
            withClockResetEnable systemClockGen resetGen enableGen $
              let ddrSlave = createDRAMBackedAxiSlave params ddrMaster'
                  (ddrMaster', weightStream', streamValid', _) =
                    weightManagementSystem ddrSlave startStream layerIdx sinkRdy
              in (weightStream', streamValid')

      let sampledWeights = sampleN totalCycles weightStream
          sampledValid = sampleN totalCycles streamValid

          -- Extract when valid is true
          validWeights = [w | (w, True) <- P.zip sampledWeights sampledValid]

      P.putStrLn "\nFirst 10 valid beats:"
      P.mapM_ (\(i, w) -> P.putStrLn $ "  Beat " P.++ show i P.++ ": " P.++ show (P.take 8 $ toList (bitCoerce w :: Vec 64 (BitVector 8))))
        $ P.take 10 $ P.zip [0::Int ..] validWeights

      -- Verify beat 1 (first Q weights) is non-zero
      let beat1 = validWeights P.!! 1
          beat1Bytes = bitCoerce beat1 :: Vec 64 (BitVector 8)
          hasNonZero = P.any (/= 0) (toList beat1Bytes)

      hasNonZero `shouldBe` True

      P.length validWeights `shouldSatisfy` (> 0)
      hasNonZero `shouldBe` True

    it "DIAGNOSTIC: verify packRowToWord round-trip" $ do
      let
          params = PARAM.decoderConst
          layer0 = head (PARAM.modelLayers params)
          mha = PARAM.multiHeadAttention layer0

          -- Get the actual Q matrix for head 0
          qHead0 = head (PARAM.headsQ mha)
          qMatrix = PARAM.wqHeadQ qHead0

          -- Get first row (the one we're comparing)
          (originalMants, originalExp) = qMatrix !! (0 :: Int)
          firstOriginalMant = head originalMants

      P.putStrLn "\n=== Original params (head 0, row 0) ==="
      P.putStrLn $ "First mantissa: " P.++ show firstOriginalMant
      P.putStrLn $ "Exponent: " P.++ show originalExp

      -- Now pack it and unpack it
      let packed = packRowToWord (originalMants, originalExp)
          unpacked = bitCoerce packed :: Vec 64 (BitVector 8)

          unpackedMant0 = bitCoerce (head unpacked) :: Signed 8
          unpackedMant1 = bitCoerce (unpacked !! (1 :: Int)) :: Signed 8

      P.putStrLn "\n=== After pack/unpack ==="
      P.putStrLn $ "Byte 0 (should be mant0=" P.++ show firstOriginalMant P.++ "): " P.++ show unpackedMant0
      P.putStrLn $ "Byte 1 (should be mant1): " P.++ show unpackedMant1

      unpackedMant0 `shouldBe` firstOriginalMant

    it "DIAGNOSTIC: verify memory addressing matches layout" $ do
      let
          params = PARAM.decoderConst
          mem = buildMemoryFromParams params
          
          layer0 = head (PARAM.modelLayers params)
          mha = PARAM.multiHeadAttention layer0
          qHead0 = head (PARAM.headsQ mha)
          (expectedMants, expectedExp) = PARAM.wqHeadQ qHead0 !! (0 :: Int)
          expectedMant0 = head expectedMants
          
      P.putStrLn "\n=== Expected Q head 0, row 0 ==="
      P.putStrLn $ "First mantissa: " P.++ show expectedMant0
      P.putStrLn $ "Exponent: " P.++ show expectedExp
      
      -- Use ABSOLUTE addressing (same as "verify absolute addressing" test)
      let modelDim = natToNum @ModelDimension
          layer0Base = calculateLayerBaseAddress 0
          rmsAttBytes = modelDim + 1
          qRow0ByteAddr = layer0Base + rmsAttBytes
          qRow0WordAddr = fromIntegral (qRow0ByteAddr `shiftR` 6) :: Int
          qRow0ByteOffset = fromIntegral (qRow0ByteAddr .&. 0x3F) :: Int
          
      P.putStrLn $ "Q head 0, row 0 at word " P.++ show qRow0WordAddr P.++ ", byte offset " P.++ show qRow0ByteOffset
      
      let packed = mem !! qRow0WordAddr
          unpacked = bitCoerce packed :: Vec 64 (BitVector 8)
          actualMant0 = bitCoerce (unpacked !! qRow0ByteOffset) :: Signed 8
          
      P.putStrLn $ "First mantissa: " P.++ show actualMant0
      
      actualMant0 `shouldBe` expectedMant0

    it "DIAGNOSTIC: verify decoder reads correct address" $ do
      let
          params = PARAM.decoderConst
          layerIdx = pure 0
          startStream = fromList $ P.replicate 5 False P.++ P.replicate 10 True P.++ P.repeat False
          sinkRdy = pure True
          totalCycles = 1000

      let (ddrMaster, weightStream, streamValid) = 
            withClockResetEnable systemClockGen resetGen enableGen $ 
              let ddrSlave = createDRAMBackedAxiSlave params ddrMaster'
                  (ddrMaster', weightStream', streamValid', _) =
                    weightManagementSystem ddrSlave startStream layerIdx sinkRdy
              in (ddrMaster', weightStream', streamValid')
      
      let sampledArValid = sampleN totalCycles (Master.arvalid ddrMaster)
          sampledArData = sampleN totalCycles (Master.ardata ddrMaster)
          sampledValid = sampleN totalCycles streamValid
          sampledWeights = sampleN totalCycles weightStream
          
          -- Find first AXI address request
          firstArRequest = DL.find fst $ P.zip sampledArValid sampledArData
          
          -- Find first valid weight beat
          firstValidWeight = DL.find fst $ P.zip sampledValid sampledWeights
          
      case firstArRequest of
        Just (_, arData) -> do
          let addr = AXITypes.araddr arData
          P.putStrLn "\n=== First AXI read request ==="
          P.putStrLn $ "Address: 0x" P.++ showHex addr ""
          P.putStrLn $ "Expected layer 0 base: 0x" P.++ showHex (calculateLayerBaseAddress 0) ""
        Nothing -> P.putStrLn "No AXI request found"
        
      case firstValidWeight of
        Just (_, weight) -> do
          let bytes = bitCoerce weight :: Vec 64 (BitVector 8)
              mant0 = bitCoerce (head bytes) :: Signed 8
          P.putStrLn "\n=== First weight data received ==="
          P.putStrLn $ "First mantissa: " P.++ show mant0
          P.putStrLn $ "First 8 bytes: " P.++ show (P.take 8 $ toList bytes)
        Nothing -> P.putStrLn "No valid weight found"
      
      True `shouldBe` True

    it "DIAGNOSTIC: verify absolute addressing" $ do
      let
          params = PARAM.decoderConst
          mem = buildMemoryFromParams params
          
          -- Calculate where the decoder will actually read for Q head 0 row 0
          layer0Base = calculateLayerBaseAddress 0
          modelDim = natToNum @ModelDimension
          rmsAttBytes = modelDim + 1  -- 65 bytes
          qHead0Row0ByteAddr = layer0Base + rmsAttBytes
          qHead0Row0WordAddr = fromIntegral (qHead0Row0ByteAddr `shiftR` 6) :: Int
          byteOffsetInWord = fromIntegral (qHead0Row0ByteAddr .&. 0x3F) :: Int

      P.putStrLn "\n=== Absolute Addressing ==="
      P.putStrLn $ "Layer 0 base byte address: 0x" P.++ showHex layer0Base ""
      P.putStrLn $ "Q head 0 row 0 byte address: 0x" P.++ showHex qHead0Row0ByteAddr ""
      P.putStrLn $ "Q head 0 row 0 word address: " P.++ show qHead0Row0WordAddr
      P.putStrLn $ "Byte offset within word: " P.++ show byteOffsetInWord
      
      let packed = mem !! qHead0Row0WordAddr
          unpacked = bitCoerce packed :: Vec 64 (BitVector 8)
          -- Account for offset!
          actualMant0 = bitCoerce (unpacked !! byteOffsetInWord) :: Signed 8
          
      P.putStrLn $ "First mantissa at byte offset " P.++ show byteOffsetInWord P.++ ": " P.++ show actualMant0
      
      let layer0 = head (PARAM.modelLayers params)
          qMat = PARAM.wqHeadQ (head (PARAM.headsQ (PARAM.multiHeadAttention layer0)))
          (expectedMants, _) = qMat !! (0 :: Int)
          expectedMant0 = head expectedMants
          
      P.putStrLn $ "Expected: " P.++ show expectedMant0
      
      actualMant0 `shouldBe` expectedMant0
      
    it "DIAGNOSTIC: check byte array construction" $ do
      let
          params = PARAM.decoderConst
          
          -- Build the byte array manually
          modelDim = natToNum @ModelDimension
          seqLen = natToNum @SequenceLength
          rotaryDim = natToNum @RotaryPositionalEmbeddingDimension
          
          embeddingBytes = 
            let vocab = PARAM.vocabularyQ (PARAM.modelEmbedding params)
            in P.concatMap rowToBytes (toList vocab)
          
          rmsFinalBytes = P.replicate (modelDim + 1) (0 :: Int)
          rotaryBytes = P.replicate (2 * seqLen * rotaryDim * 4) (0 :: Int)
          
          layer0 = head (PARAM.modelLayers params)
          mha = PARAM.multiHeadAttention layer0
          
          rmsAttBytes = P.replicate (modelDim + 1) (0 :: Int)
          
          qHead0 = head (PARAM.headsQ mha)
          qMat = PARAM.wqHeadQ qHead0
          (firstRowMants, _firstRowExp) = qMat !! (0 :: Int)
          expectedMant0 = head firstRowMants
          
          qHead0Bytes = P.concatMap rowToBytes (toList qMat)
          
      P.putStrLn "\n=== Byte Array Construction ==="
      P.putStrLn $ "Embedding bytes: " P.++ show (P.length embeddingBytes)
      P.putStrLn $ "RMS final bytes: " P.++ show (P.length rmsFinalBytes)
      P.putStrLn $ "Rotary bytes: " P.++ show (P.length rotaryBytes)
      P.putStrLn $ "Layer 0 base should be at: " P.++ show (P.length embeddingBytes + P.length rmsFinalBytes + P.length rotaryBytes)
      P.putStrLn $ "Expected: 0xc241 = " P.++ show (0xc241 :: Int)
      
      P.putStrLn $ "\nRMS Att bytes: " P.++ show (P.length rmsAttBytes)
      P.putStrLn $ "Q head 0 bytes: " P.++ show (P.length qHead0Bytes)
      P.putStrLn $ "First Q byte should be: " P.++ show expectedMant0
      P.putStrLn $ "Actual first Q byte: " P.++ show (P.head qHead0Bytes)
      
      P.head qHead0Bytes `shouldBe` pack expectedMant0

rowToBytes :: (Vec n Mantissa, Exponent) -> [BitVector 8]
rowToBytes (mantissas, expon) =
  let
    mantBytes :: [BitVector 8]
    mantBytes = P.map pack (toList mantissas)
    expByte :: BitVector 8
    expByte = resize (pack expon)
  in mantBytes P.++ [expByte]
