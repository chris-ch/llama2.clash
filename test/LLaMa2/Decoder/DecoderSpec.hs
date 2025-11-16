module LLaMa2.Decoder.DecoderSpec (spec) where

import Clash.Prelude
import qualified Data.List as DL
import Test.Hspec
import qualified Prelude as P

import qualified LLaMa2.Decoder.Decoder as Decoder
import LLaMa2.Types.LayerData (Token, Temperature, Seed, LayerData(..))
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.ParamsPlaceholder as PARAM
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import Simulation.Parameters (DecoderParameters)

-- Helper: compute L2 norm
normVec :: Vec n FixedPoint -> Float
normVec v = sqrt (sum squares)
  where
    floats = P.map (realToFrac . toRational) (toList v) :: [Float]
    squares = P.map (\x -> x * x) floats

-- Expected values from Python reference
data ExpectedNorms = ExpectedNorms
  { attnNorm :: Double
  , ffnNorm :: Double
  } deriving (Show)

-- Token 0 (BOS = 1) expected norms per layer
token0Expected :: [(Int, ExpectedNorms)]
token0Expected =
  [ (0, ExpectedNorms 3.0385 8.2357)
  , (1, ExpectedNorms 9.1765 10.3825)
  , (2, ExpectedNorms 11.2580 12.2522)
  , (3, ExpectedNorms 14.7369 16.0392)
  , (4, ExpectedNorms 16.4430 18.7545)
  ]

-- Token 1 (320) expected norms per layer  
token1Expected :: [(Int, ExpectedNorms)]
token1Expected =
  [ (0, ExpectedNorms 2.7743 3.8518)
  , (1, ExpectedNorms 4.5273 5.3427)
  , (2, ExpectedNorms 6.4987 6.8089)
  , (3, ExpectedNorms 8.5103 9.4203)
  , (4, ExpectedNorms 9.9178 13.6559)
  ]

type DecoderInputState = (Token, [Token], Bool)

-- | Bundled outputs matching Main.hs exactly
bundledOutputs
  :: Signal System (Token, Bool, Temperature, Seed)
  -> ( Signal System (Token, Bool)
     , Decoder.DecoderIntrospection System
     )
bundledOutputs bundledInputs =
  (bundle (tokenOut, validOut), introspection)
 where
  (token, isValid, temperature, seed) = unbundle bundledInputs

  params :: DecoderParameters
  params = PARAM.decoderConst

  -- Create DDR simulator
  dramSlaveIn = exposeClockResetEnable
    (DRAMSlave.createDRAMBackedAxiSlave params ddrMaster)
    systemClockGen
    resetGen
    enableGen

  -- Decoder with AXI feedback loop
  (ddrMaster, tokenOut, validOut, introspection) =
    exposeClockResetEnable
      (Decoder.decoder dramSlaveIn params token isValid temperature seed)
      systemClockGen
      resetGen
      enableGen

spec :: Spec
spec = do
  describe "Decoder - Multi-Token State Pollution Detection" $ do
    it "detects state pollution between tokens by tracking all layer norms" $ do
        let
            promptTokens = [1, 320] :: [Token]
            temperature = 0.0 :: FixedPoint
            seed = 123 :: Seed
            maxCycles = 12_000
            
            -- Autoregressive state management (from Main.hs)
            (firstToken, restPrompt) = case promptTokens of
              (t:ts) -> (t, ts)
              []     -> (1, [])
            
            advanceState (current, remPrompt, usingPrompt) (isReady, sampled)
              | not isReady = (current, remPrompt, usingPrompt)
              | otherwise   = case remPrompt of
                                (p:ps) -> (p, ps, True)
                                []     -> (sampled, [], False)
            
            -- Build signals
            inputSignals = fromList (DL.zip4 inputTokens inputValidFlags (P.repeat temperature) (P.repeat seed))
            
            (coreOutputsSignal, introspection) = bundledOutputs inputSignals
            
            -- Sample core outputs
            coreOutputs :: [(Token, Bool)]
            coreOutputs = sampleN maxCycles coreOutputsSignal
            
            (outputTokens, readyFlags) = P.unzip coreOutputs
            
            -- Derive evolving state
            states :: [DecoderInputState]
            states = P.scanl advanceState (firstToken, restPrompt, True)
                        (P.zip (P.drop 1 readyFlags) (P.drop 1 outputTokens))
            
            inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
            inputValidFlags = True : [ usePrompt | (_, _, usePrompt) <- states ]
            
            -- Sample introspection signals
            cycles = [0 .. maxCycles - 1]
            layerIndices = P.take maxCycles $ P.map fromIntegral $ sampleN maxCycles (Decoder.layerIndex introspection)
            attnDones = P.take maxCycles $ sampleN maxCycles (Decoder.attnDone introspection)
            ffnDones = P.take maxCycles $ sampleN maxCycles (Decoder.ffnDone introspection)
            layerDataList = P.take maxCycles $ sampleN maxCycles (Decoder.layerData introspection)
            
            -- Extract completion events with norms
            extractCompletions :: Int -> Int -> [(Int, Int, Double, Double)]
            extractCompletions startCycle endCycle =
                let attnEvents = 
                        [ (cycle', li, normVec (attentionOutput ld))
                        | (cycle', li, attnDone, ld) <- DL.zip4 cycles layerIndices attnDones layerDataList
                        , cycle' >= startCycle && cycle' < endCycle
                        , attnDone
                        ]
                    
                    ffnEvents =
                        [ (cycle', li, normVec (feedForwardOutput ld))
                        | (cycle', li, ffnDone, ld) <- DL.zip4 cycles layerIndices ffnDones layerDataList
                        , cycle' >= startCycle && cycle' < endCycle
                        , ffnDone
                        ]
                in [(li, attnCycle, attnN, ffnN) 
                   | li <- [0..4]
                   , let attnCycle = case DL.find (\(_, l, _) -> l == li) attnEvents of
                           Just (c, _, _) -> c
                           Nothing -> -1
                   , let attnN = case DL.find (\(_, l, _) -> l == li) attnEvents of
                           Just (_, _, n) -> realToFrac n :: Double
                           Nothing -> -1.0
                   , let ffnN = case DL.find (\(_, l, _) -> l == li) ffnEvents of
                           Just (_, _, n) -> realToFrac n :: Double
                           Nothing -> -1.0
                   ]
            
            -- Find when tokens actually complete
            tokenCompletions = [ (i, tok) | (i, (tok, ready)) <- P.zip cycles coreOutputs, ready ]
            
            token0End = case DL.find (\(_, tok) -> tok == 1) tokenCompletions of
                Just (c, _) -> c
                Nothing -> 10_000  -- fallback
            
            token1Start = token0End + 1
            
            token0Events = extractCompletions 0 token0End
            token1Events = extractCompletions token1Start maxCycles
            
            -- Print detailed comparison
            printTokenResults tokenNum events expected = do
                P.putStrLn $ "\n=== TOKEN " P.++ show tokenNum P.++ " ==="
                P.mapM_ (\(li, cycle', attnN, ffnN) -> do
                    let exp' = DL.lookup li expected
                    case exp' of
                        Just (ExpectedNorms expAttn expFFN) -> do
                            let attnErr = abs (attnN - expAttn)
                            let ffnErr = abs (ffnN - expFFN)
                            let attnStatus = if attnErr < 0.01 then "✓" else "✗ FAIL"
                            let ffnStatus = if ffnErr < 0.01 then "✓" else "✗ FAIL"
                            P.putStrLn $ "Layer " P.++ show li P.++ " (cycle " P.++ show cycle' P.++ "):"
                            P.putStrLn $ "  Attn: " P.++ show attnN P.++ " (exp: " P.++ show expAttn P.++ ", err: " P.++ show attnErr P.++ ") " P.++ attnStatus
                            P.putStrLn $ "  FFN:  " P.++ show ffnN P.++ " (exp: " P.++ show expFFN P.++ ", err: " P.++ show ffnErr P.++ ") " P.++ ffnStatus
                        Nothing -> do
                            P.putStrLn $ "Layer " P.++ show li P.++ ": NO EXPECTED VALUES"
                            P.putStrLn $ "  Attn: " P.++ show attnN
                            P.putStrLn $ "  FFN:  " P.++ show ffnN
                    ) events
        
        printTokenResults (0 :: Int) token0Events token0Expected
        printTokenResults (1 :: Int) token1Events token1Expected
        
        -- Focused assertion: Token 1 Layer 0 attention norm
        let token1Layer0Attn = case DL.find (\(li, _, _, _) -> li == 0) token1Events of
                Just (_, _, attnN, _) -> attnN
                Nothing -> error "No layer 0 attention event found for token 1"
        
        let expectedToken1Layer0Attn = case DL.lookup 0 token1Expected of
                Just (ExpectedNorms attnN _) -> attnN
                Nothing -> error "No expected value for token 1 layer 0"
        
        P.putStrLn "\n=== CRITICAL BUG CHECK ==="
        P.putStrLn "Token 1 Layer 0 Attention Norm:"
        P.putStrLn $ "  Actual:   " P.++ show token1Layer0Attn
        P.putStrLn $ "  Expected: " P.++ show expectedToken1Layer0Attn
        P.putStrLn $ "  Error:    " P.++ show (abs (token1Layer0Attn - expectedToken1Layer0Attn))
        
        abs (token1Layer0Attn - expectedToken1Layer0Attn) `shouldSatisfy` (< 0.01)
