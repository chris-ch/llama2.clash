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
import LLaMa2.Types.LayerData (Token, Temperature, Seed)
import LLaMa2.Types.ModelConfig (NumLayers, VocabularySize)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified Simulation.ParamsPlaceholder as PARAM
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import Simulation.Parameters (DecoderParameters)

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
      (Decoder.decoder (pure 0) dramSlaveIn kvDramSlaves token isValid (pure False) temperature seed)
      systemClockGen
      resetGen
      enableGen

spec :: Spec
spec = do
  -- -------------------------------------------------------------------------
  -- ffnOut0 reference check: compare layer-by-layer FFN output[0] against
  -- the Phase 1 (wide-bus) baseline captured at --temperature 0 --seed 123
  -- prompt "Hi" (tokens [1, 320]).
  -- Runs under model-260k only; nano model has different weight values.
  -- -------------------------------------------------------------------------
  describe "Decoder - ffnOut0 reference check (model-260k)" $ do
#ifdef MODEL_NANO
    it "skipped: needs model-260k" $ do
      pendingWith "Run with make test-full (model-260k)"
#else
    it "ffnOut0 matches Phase 1 baseline: token 0 layers 0-1" $ do
      let
        promptTokens = [1] :: [Token]
        temperature  = 0.0 :: FixedPoint
        seed         = 123 :: Seed
        maxCycles    = 60_000

        (firstToken, restPrompt) = case promptTokens of
          (t:ts) -> (t, ts)
          []     -> (1, [])

        advanceState (current, remPrompt, usingPrompt) (isReady, sampled)
          | not isReady = (current, remPrompt, usingPrompt)
          | otherwise   = case remPrompt of
                            (p:ps) -> (p, ps, True)
                            []     -> (sampled, [], False)

        inputSignals = fromList (DL.zip4 inputTokens inputValidFlags
                                   (P.repeat temperature) (P.repeat seed))

        (coreOutputsSignal, introspection) = bundledOutputs inputSignals

        (tokenSig, validSig) = unbundle coreOutputsSignal

        allSampled = sampleN maxCycles $ bundle
          ( tokenSig
          , validSig
          , Decoder.layerIndex introspection
          , Decoder.layerDone  introspection
          , Decoder.ffnOut0    introspection
          )

        outputTokens = [tok | (tok, _, _, _, _) <- allSampled]
        readyFlags   = [v   | (_, v, _, _, _)   <- allSampled]
        layerIndices = [fromIntegral li | (_, _, li, _, _) <- allSampled] :: [Int]
        layerDones   = [d   | (_, _, _, d, _)   <- allSampled]
        ffnOut0s     = [fv  | (_, _, _, _, fv)  <- allSampled]

        states :: [DecoderInputState]
        states = P.scanl advanceState (firstToken, restPrompt, True)
                   (P.zip (P.drop 1 readyFlags) (P.drop 1 outputTokens))

        inputTokens     = firstToken : [tok | (tok, _, _) <- states]
        inputValidFlags = True       : [u   | (_, _, u)   <- states]

        -- (layerIdx, ffnOut0) for every layerDone pulse, in order
        doneEvents :: [(Int, Double)]
        doneEvents =
          [ (li, realToFrac fv)
          | (li, done, fv) <- P.zip3 layerIndices layerDones ffnOut0s
          , done
          ]

        -- Reference values from Phase 1 simulation (token 0 = BOS)
        expected :: [(Int, Double)]
        expected =
          [ (0,  0.34451)  -- token 0, layer 0
          , (1,  0.74667)  -- token 0, layer 1
          ]

        tol = 0.001 :: Double

      P.putStrLn $ "\n[ffnOut0 test] done events: " P.++ show (P.take 5 doneEvents)
      hFlush stdout

      P.length doneEvents `shouldSatisfy` (>= 2)
      P.mapM_ (\((eli, ev), (ali, av)) -> do
          P.putStrLn $ "  layer " P.++ show eli
            P.++ " expected=" P.++ show ev
            P.++ " actual=" P.++ show av
            P.++ if abs (ev - av) < tol then " OK" else " FAIL"
          hFlush stdout
          eli `shouldBe` ali
          abs (ev - av) `shouldSatisfy` (< tol)
        ) (P.zip expected doneEvents)
#endif

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

            -- Single bundled sampleN
            (tokenSig, validSig) = unbundle coreOutputsSignal

            allSampled = sampleN maxCycles $ bundle
              ( tokenSig
              , validSig
              , Decoder.layerIndex introspection
              , Decoder.layerDone introspection
              )

            outputTokens  = [tok | (tok, _, _, _) <- allSampled]
            readyFlags    = [v   | (_, v, _, _)   <- allSampled]
            cycles        = [0 .. maxCycles - 1]
            layerIndices  = P.map fromIntegral [li | (_, _, li, _) <- allSampled]
            layerDones    = [d   | (_, _, _, d)   <- allSampled]

            -- Derive evolving state
            states :: [DecoderInputState]
            states = P.scanl advanceState (firstToken, restPrompt, True)
                        (P.zip (P.drop 1 readyFlags) (P.drop 1 outputTokens))

            inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
            inputValidFlags = True : [ usePrompt | (_, _, usePrompt) <- states ]

            -- Extract completion events: (layerIdx, layerDoneCycle)
            -- -1 sentinel means the event was not found in this window.
            extractCompletions :: Int -> Int -> [(Int, Int)]
            extractCompletions startCycle endCycle =
                let doneEvents =
                        [ (cycle', li)
                        | (cycle', li, done) <- DL.zip3 cycles layerIndices layerDones
                        , cycle' >= startCycle && cycle' < endCycle
                        , done
                        ]
                in [ (li, doneCycle)
                   | li <- [0..natToNum @NumLayers - 1 :: Int]
                   , let doneCycle = maybe (-1) P.fst (DL.find (\(_, l) -> l == li) doneEvents)
                   ]

            -- Print cycle info per token
            printTokenResults tokenNum events = do
                P.putStrLn $ "\n=== TOKEN " P.++ show tokenNum P.++ " ==="
                P.mapM_ (\(li, doneCycle) ->
                    P.putStrLn $ "Layer " P.++ show li
                      P.++ " done@" P.++ show doneCycle
                    ) events

        -- Eagerly find token0End before processing
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

        result <- try @SomeException $ do
            printTokenResults (0 :: Int) token0Events
            hFlush stdout
            printTokenResults (1 :: Int) token1Events
            hFlush stdout

#ifdef MODEL_NANO
            -- End-to-end checks: every layer must complete, tokens must be produced.
            let nLayers    = natToNum @NumLayers :: Int
                vocabSize  = natToNum @VocabularySize :: Token
                sampledTokens = [ tok | (tok, rdy) <- P.zip outputTokens readyFlags, rdy ]
                allLayerDoneFired = P.all (\(_, doneCycle) -> doneCycle >= 0)

            -- Every layer (0 .. NumLayers-1) must have fired layerDone for token 0
            P.length token0Events `shouldBe` nLayers
            token0Events `shouldSatisfy` allLayerDoneFired

            -- Same for token 1 (second prompt token processed after ready pulse)
            P.length token1Events `shouldBe` nLayers
            token1Events `shouldSatisfy` allLayerDoneFired

            -- At least one output token must have been produced (readyPulse fired)
            sampledTokens `shouldSatisfy` (not . P.null)

            -- Every sampled token must be a valid vocabulary index
            sampledTokens `shouldSatisfy` P.all (< vocabSize)

            P.putStrLn $ "\n=== SAMPLED TOKENS ==="
            P.mapM_ (\t -> P.putStrLn $ "  " P.++ show t) sampledTokens
            hFlush stdout
#endif

        case result of
            Right () -> pure ()
            Left e   -> do
                P.putStrLn $ "\n[EXCEPTION CAUGHT]: " P.++ show e
                hFlush stdout
                expectationFailure $ show e
#endif

  -- -------------------------------------------------------------------------
  -- Multi-token autoregressive loop (Task 3)
  -- Runs model-nano with BOS prompt, verifies 3 tokens are generated and that
  -- all layers complete for each token (KV cache + slot-0 copy consistency).
  -- -------------------------------------------------------------------------
  describe "Decoder - multi-token autoregressive loop (model-nano)" $ do
#ifndef MODEL_NANO
    it "skipped: run with -f model-nano" $ do
      pendingWith "Use -f model-nano for multi-token autoregressive test"
#else
    it "generates 3 tokens with all layers completing for each" $ do
      let
        promptTokens = [1] :: [Token]  -- BOS only; model generates the rest
        temperature  = 0.0 :: FixedPoint
        seed         = 123 :: Seed
        maxCycles    = 100_000
        nLayers      = natToNum @NumLayers :: Int
        vocabSize    = natToNum @VocabularySize :: Token

        (firstToken, restPrompt) = case promptTokens of
          (t:ts) -> (t, ts)
          []     -> (1, [])

        advanceState (current, remPrompt, usingPrompt) (isReady, sampled)
          | not isReady = (current, remPrompt, usingPrompt)
          | otherwise   = case remPrompt of
                            (p:ps) -> (p, ps, True)
                            []     -> (sampled, [], False)

        inputSignals = fromList (DL.zip4 inputTokens inputValidFlags
                                   (P.repeat temperature) (P.repeat seed))

        (coreOutputsSignal, introspection) = bundledOutputs inputSignals
        (tokenSig, validSig)               = unbundle coreOutputsSignal

        allSampled = sampleN maxCycles $ bundle
          ( tokenSig
          , validSig
          , Decoder.layerIndex introspection
          , Decoder.layerDone  introspection
          )

        outputTokens = [tok | (tok, _, _, _) <- allSampled]
        readyFlags   = [v   | (_, v, _, _)   <- allSampled]
        layerIndices = P.map fromIntegral [li | (_, _, li, _) <- allSampled] :: [Int]
        layerDones   = [d   | (_, _, _, d)   <- allSampled]
        cycles       = [0 .. maxCycles - 1] :: [Int]

        states :: [DecoderInputState]
        states = P.scanl advanceState (firstToken, restPrompt, True)
                   (P.zip (P.drop 1 readyFlags) (P.drop 1 outputTokens))

        inputTokens     = firstToken : [tok | (tok, _, _) <- states]
        inputValidFlags = True       : [u   | (_, _, u)   <- states]

        -- Collect all layerDone events as (cycle, layerIdx)
        allDoneEvents :: [(Int, Int)]
        allDoneEvents =
          [ (c, li)
          | (c, li, done) <- DL.zip3 cycles layerIndices layerDones
          , done
          ]

        -- Find cycle boundaries where readyPulse fires
        readyPulseCycles :: [Int]
        readyPulseCycles = [c | (c, r) <- DL.zip cycles readyFlags, r]

        sampledTokens :: [Token]
        sampledTokens = [tok | (tok, rdy) <- P.zip outputTokens readyFlags, rdy]

      P.putStrLn $ "\n[multi-token] readyPulse cycles: " P.++ show (P.take 5 readyPulseCycles)
      P.putStrLn $ "[multi-token] sampled tokens:    " P.++ show (P.take 5 sampledTokens)
      P.putStrLn $ "[multi-token] total layerDones:  " P.++ show (P.length allDoneEvents)
      hFlush stdout

      -- Must produce at least 3 tokens
      P.length sampledTokens `shouldSatisfy` (>= 3)

      -- Every sampled token must be a valid vocabulary index
      sampledTokens `shouldSatisfy` P.all (< vocabSize)

      -- For each of the first 3 tokens, all NumLayers layers must have fired layerDone.
      -- We slice allDoneEvents by the readyPulse boundaries.
      let tokenBoundaries = P.zip (0 : readyPulseCycles) readyPulseCycles
          firstThreeBounds = P.take 3 tokenBoundaries

      P.mapM_ (\(start, end) -> do
          let eventsInWindow =
                [ li | (c, li) <- allDoneEvents, c >= start, c < end ]
          -- All nLayers layers must appear
          P.length eventsInWindow `shouldBe` nLayers
          DL.sort eventsInWindow  `shouldBe` [0 .. nLayers - 1]
        ) firstThreeBounds
#endif
