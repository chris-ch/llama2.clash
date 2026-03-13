{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -Wno-unused-imports -Wno-unused-top-binds #-}
module LLaMa2.Decoder.DecoderSpec (spec) where

import Clash.Prelude
import qualified Data.List as DL
import Test.Hspec
import qualified Prelude as P
import Control.Exception (try, SomeException)
import System.IO (hFlush, stdout)

import qualified LLaMa2.Decoder.Decoder as Decoder
import LLaMa2.Types.LayerData (Token, Temperature, Seed, LayerData(..))
import LLaMa2.Types.ModelConfig (NumLayers, VocabularySize)
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
    (DRAMSlave.createDRAMBackedAxiSlave (pure 0) params ddrMaster)
    systemClockGen
    resetGen
    enableGen

  -- KV cache DRAM slaves (lazy circular dependency with decoder's KV masters)
  kvDramSlaves = exposeClockResetEnable
    (map (DRAMSlave.createKVCacheDRAMSlave (pure 0)) kvMasters)
    systemClockGen
    resetGen
    enableGen

  -- Decoder with AXI feedback loop
  (ddrMaster, kvMasters, tokenOut, validOut, introspection) =
    exposeClockResetEnable
      (Decoder.decoder (pure 0) dramSlaveIn kvDramSlaves token isValid temperature seed)
      systemClockGen
      resetGen
      enableGen

spec :: Spec
spec = do
  describe "Decoder - Multi-Token State Pollution Detection" $ do
#ifndef MODEL_NANO
    it "skipped: run with -f model-nano for fast decoder simulation" $ do
      pendingWith "Use -f model-nano -f -model-260k for a fast decoder test"
#else
    it "detects state pollution between tokens by tracking all layer norms" $ do
        let
            promptTokens = [1, 320] :: [Token]
            temperature = 0.0 :: FixedPoint
            seed = 123 :: Seed
#ifdef MODEL_NANO
            maxCycles = 10_000
#else
            maxCycles = 60_000
#endif
            
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

            -- Single bundled sampleN: allows per-cycle circuit state to be GC'd
            -- (5 separate sampleN calls would hold the full circuit state for all
            --  maxCycles cycles alive simultaneously via shared thunks, causing OOM)
            (tokenSig, validSig) = unbundle coreOutputsSignal

            allSampled = sampleN maxCycles $ bundle
              ( tokenSig
              , validSig
              , Decoder.layerIndex introspection
              , Decoder.attnDone introspection
              , Decoder.ffnDone introspection
              , Decoder.layerData introspection
              )

            outputTokens  = [tok | (tok, _, _, _, _, _) <- allSampled]
            readyFlags    = [v   | (_, v, _, _, _, _)   <- allSampled]
            cycles        = [0 .. maxCycles - 1]
            layerIndices  = P.map fromIntegral [li | (_, _, li, _, _, _) <- allSampled]
            attnDones     = [d   | (_, _, _, d, _, _)   <- allSampled]
            ffnDones      = [d   | (_, _, _, _, d, _)   <- allSampled]
            layerDataList = [ld  | (_, _, _, _, _, ld)  <- allSampled]

            -- Derive evolving state
            states :: [DecoderInputState]
            states = P.scanl advanceState (firstToken, restPrompt, True)
                        (P.zip (P.drop 1 readyFlags) (P.drop 1 outputTokens))

            inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
            inputValidFlags = True : [ usePrompt | (_, _, usePrompt) <- states ]
            
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
                   | li <- [0..natToNum @NumLayers - 1 :: Int]
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
        
        -- Eagerly find token0End before any LayerData is forced
        P.putStrLn $ "[DBG] Searching for token0End in " P.++ show maxCycles P.++ " cycles..."
        hFlush stdout
        token0EndResult <- try @SomeException $ do
            let idx = DL.findIndex id readyFlags
            let v = case idx of
                      Just i  -> i + 1
                      Nothing -> maxCycles `P.div` 2
            P.putStrLn $ "[DBG] token0End=" P.++ show v
            hFlush stdout
            return v
        token0End <- case token0EndResult of
            Right v -> return v
            Left e  -> do
                P.putStrLn $ "[DBG EXCEPTION in findIndex]: " P.++ show e
                hFlush stdout
                expectationFailure $ "findIndex threw: " P.++ show e
                return (maxCycles `P.div` 2)

        let token1Start = token0End

        let token0Events = extractCompletions 0 token0End
            token1Events = extractCompletions token1Start maxCycles

        -- Wrap everything in try to catch P.error / exceptions
        result <- try @SomeException $ do
            P.putStrLn "[DBG] About to evaluate token0Events length..."
            hFlush stdout
            let n0 = P.length token0Events
            P.putStrLn $ "[DBG] token0Events has " P.++ show n0 P.++ " entries"
            hFlush stdout

            printTokenResults (0 :: Int) token0Events token0Expected
            hFlush stdout

            P.putStrLn "[DBG] token 0 done, evaluating token1Events..."
            hFlush stdout
            let n1 = P.length token1Events
            P.putStrLn $ "[DBG] token1Events has " P.++ show n1 P.++ " entries"
            hFlush stdout

            printTokenResults (1 :: Int) token1Events token1Expected
            hFlush stdout

#ifdef MODEL_NANO
            -- End-to-end checks: every layer must complete, tokens must be produced.
            let nLayers    = natToNum @NumLayers :: Int
                vocabSize  = natToNum @VocabularySize :: Token
                -- Tokens emitted when readyPulse is high
                sampledTokens = [ tok | (tok, rdy) <- P.zip outputTokens readyFlags, rdy ]
                -- Sentinel -1 / -1.0 means event was not found in this window
                allAttnFired = P.all (\(_, attnCycle, _, _) -> attnCycle >= 0)
                allFfnFired  = P.all (\(_, _, _, ffnN)      -> ffnN /= -1.0)

            -- Every layer (0 .. NumLayers-1) must have fired attnDone for token 0
            P.length token0Events `shouldBe` nLayers
            token0Events `shouldSatisfy` allAttnFired
            token0Events `shouldSatisfy` allFfnFired

            -- Same for token 1 (second prompt token processed after ready pulse)
            P.length token1Events `shouldBe` nLayers
            token1Events `shouldSatisfy` allAttnFired
            token1Events `shouldSatisfy` allFfnFired

            -- At least one output token must have been produced (readyPulse fired)
            sampledTokens `shouldSatisfy` (not . P.null)

            -- Every sampled token must be a valid vocabulary index
            sampledTokens `shouldSatisfy` P.all (< vocabSize)

            P.putStrLn $ "\n=== SAMPLED TOKENS ==="
            P.mapM_ (\t -> P.putStrLn $ "  " P.++ show t) sampledTokens
            hFlush stdout
#else
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
            hFlush stdout

            abs (token1Layer0Attn - expectedToken1Layer0Attn) `shouldSatisfy` (< 0.01)
#endif

        case result of
            Right () -> pure ()
            Left e   -> do
                P.putStrLn $ "\n[EXCEPTION CAUGHT]: " P.++ show e
                hFlush stdout
                expectationFailure $ show e
#endif
