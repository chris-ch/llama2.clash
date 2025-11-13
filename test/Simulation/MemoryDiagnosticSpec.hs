-- test/Simulation/MemoryDiagnosticSpec.hs
module Simulation.MemoryDiagnosticSpec (spec) where

import Test.Hspec
import qualified Prelude as P
import Clash.Prelude
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified Simulation.ParamsPlaceholder as PARAM
import LLaMa2.Memory.WeightLoader (calculateLayerBaseAddress)
import qualified Simulation.Parameters as PARAM
import LLaMa2.Types.ModelConfig (ModelDimension, VocabularySize)

spec :: Spec
spec = describe "Memory Layout Diagnostic" $ do
    it "verify what's at address 33280 (where weight loader reads)" $ do
        let params = PARAM.decoderConst
            mem = DRAM.buildMemoryFromParams params
            
            -- Where weight loader ACTUALLY reads
            actualBase = calculateLayerBaseAddress 0
            
            -- Extract bytes from that address
            wordIdx = fromIntegral (actualBase `div` 64) :: Int
            byteOffset = fromIntegral (actualBase `mod` 64) :: Int
            
            word = mem !! wordIdx
            bytes = bitCoerce word :: Vec 64 (BitVector 8)
            firstByte = bitCoerce (bytes !! byteOffset) :: Signed 8
            
        putStrLn "\n=== What's at weight loader's address ==="
        putStrLn $ "Weight loader reads from: " P.++ show actualBase
        putStrLn $ "First byte at that address: " P.++ show firstByte
        putStrLn "Expected (first Q mantissa): 11"
        
        firstByte `shouldBe` 11
        
    it "verify layerBytes is not empty" $ do
        let params = PARAM.decoderConst
            layers = toList (PARAM.modelLayers params)
            
        putStrLn "\n=== Layer Bytes Generation ==="
        putStrLn $ "Number of layers: " P.++ show (P.length layers)
        
        -- Check first layer
        let layer0 = P.head layers
            mha = PARAM.multiHeadAttention layer0
            
        -- Get Q bytes for first head
        let qHead0 = head (PARAM.headsQ mha)
            qMat = PARAM.wqHeadQ qHead0
            qRow0 = qMat !! (0 :: Int)
            
        putStrLn $ "First Q row from params: " P.++ show qRow0
        
        -- Now try to generate bytes from it
        let (mants, exp') = qRow0
            mantBytes = P.map pack (toList mants)
            expByte = resize (pack exp')
            allBytes = mantBytes P.++ [expByte]
            
        putStrLn $ "Generated " P.++ show (P.length allBytes) P.++ " bytes from first Q row"
        putStrLn $ "First byte value: " P.++ show (P.head allBytes)
        
        -- Verify first byte is 11
        bitCoerce (P.head allBytes) `shouldBe` (11 :: Signed 8)

    it "verify embedding size matches weight loader calculation" $ do
        let params = PARAM.decoderConst
            vocab = PARAM.vocabularyQ (PARAM.modelEmbedding params)
            
            -- How buildMemoryFromParams calculates it
            embeddingBytes = P.concatMap DRAM.rowToBytes (toList vocab)
            actualEmbeddingSize = P.length embeddingBytes
            
            -- How weight loader calculates it
            vocabSize = natToNum @VocabularySize
            modelDim = natToNum @ModelDimension
            expectedEmbeddingSize = vocabSize * (modelDim + 1)
            
        putStrLn "\n=== Embedding Size Verification ==="
        putStrLn $ "Actual embedding bytes: " P.++ show actualEmbeddingSize
        putStrLn $ "Weight loader expects: " P.++ show expectedEmbeddingSize
        putStrLn $ "VocabSize: " P.++ show vocabSize
        putStrLn $ "ModelDimension: " P.++ show modelDim
        
        actualEmbeddingSize `shouldBe` expectedEmbeddingSize

    it "verify layerBytes produces data at correct offset" $ do
        let params = PARAM.decoderConst
            
            -- Generate the full byte array
            embeddingBytes = P.concatMap DRAM.rowToBytes (toList (PARAM.vocabularyQ (PARAM.modelEmbedding params)))
            layerBytes = P.concatMap DRAM.layerToBytes (toList (PARAM.modelLayers params))
            
            allBytes = embeddingBytes P.++ layerBytes
            
        putStrLn "\n=== Full Byte Array Check ==="
        putStrLn $ "Embedding bytes: " P.++ show (P.length embeddingBytes)
        putStrLn $ "Layer bytes: " P.++ show (P.length layerBytes)
        putStrLn $ "Total bytes: " P.++ show (P.length allBytes)
        
        -- Check what's at position 33280
        if P.length allBytes > 33280 then do
            let byteAt33280 = allBytes P.!! 33280
            putStrLn $ "Byte at position 33280: " P.++ show (bitCoerce byteAt33280 :: Signed 8)
            bitCoerce byteAt33280 `shouldBe` (11 :: Signed 8)
        else
            expectationFailure $ "allBytes only has " P.++ show (P.length allBytes) P.++ " bytes!"

    it "verify first Q byte is at position 33345" $ do
        let params = PARAM.decoderConst
            embeddingBytes = P.concatMap DRAM.rowToBytes (toList (PARAM.vocabularyQ (PARAM.modelEmbedding params)))
            layerBytes = P.concatMap DRAM.layerToBytes (toList (PARAM.modelLayers params))
            allBytes = embeddingBytes P.++ layerBytes
            
            modelDim = natToNum @ModelDimension
            rmsAttSize = modelDim + 1
            firstQBytePos = 33280 + rmsAttSize  -- After RmsAtt
            
        putStrLn "\n=== First Q Byte Position ==="
        putStrLn $ "RmsAtt size: " P.++ show rmsAttSize
        putStrLn $ "First Q byte should be at: " P.++ show firstQBytePos
        
        if P.length allBytes > firstQBytePos then do
            let byteAtFirstQ = allBytes P.!! firstQBytePos
            putStrLn $ "Byte at position " P.++ show firstQBytePos P.++ ": " P.++ show (bitCoerce byteAtFirstQ :: Signed 8)
            bitCoerce byteAtFirstQ `shouldBe` (11 :: Signed 8)
        else
            expectationFailure "Not enough bytes!"
