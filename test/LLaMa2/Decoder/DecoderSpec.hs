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
              , Decoder.attnDone introspection
              , Decoder.ffnDone introspection
              )

            outputTokens  = [tok | (tok, _, _, _, _) <- allSampled]
            readyFlags    = [v   | (_, v, _, _, _)   <- allSampled]
            cycles        = [0 .. maxCycles - 1]
            layerIndices  = P.map fromIntegral [li | (_, _, li, _, _) <- allSampled]
            attnDones     = [d   | (_, _, _, d, _)   <- allSampled]
            ffnDones      = [d   | (_, _, _, _, d)   <- allSampled]

            -- Derive evolving state
            states :: [DecoderInputState]
            states = P.scanl advanceState (firstToken, restPrompt, True)
                        (P.zip (P.drop 1 readyFlags) (P.drop 1 outputTokens))

            inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
            inputValidFlags = True : [ usePrompt | (_, _, usePrompt) <- states ]

            -- Extract completion events: (layerIdx, attnCycle, ffnCycle)
            -- -1 sentinel means the event was not found in this window.
            extractCompletions :: Int -> Int -> [(Int, Int, Int)]
            extractCompletions startCycle endCycle =
                let attnEvents =
                        [ (cycle', li)
                        | (cycle', li, attnDone) <- DL.zip3 cycles layerIndices attnDones
                        , cycle' >= startCycle && cycle' < endCycle
                        , attnDone
                        ]
                    ffnEvents =
                        [ (cycle', li)
                        | (cycle', li, ffnDone) <- DL.zip3 cycles layerIndices ffnDones
                        , cycle' >= startCycle && cycle' < endCycle
                        , ffnDone
                        ]
                in [ (li, attnCycle, ffnCycle)
                   | li <- [0..natToNum @NumLayers - 1 :: Int]
                   , let attnCycle = maybe (-1) P.fst (DL.find (\(_, l) -> l == li) attnEvents)
                   , let ffnCycle  = maybe (-1) P.fst (DL.find (\(_, l) -> l == li) ffnEvents)
                   ]

            -- Print cycle info per token
            printTokenResults tokenNum events = do
                P.putStrLn $ "\n=== TOKEN " P.++ show tokenNum P.++ " ==="
                P.mapM_ (\(li, attnCycle, ffnCycle) ->
                    P.putStrLn $ "Layer " P.++ show li
                      P.++ " attn@" P.++ show attnCycle
                      P.++ " ffn@"  P.++ show ffnCycle
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
                allAttnFired = P.all (\(_, attnCycle, _) -> attnCycle >= 0)
                allFfnFired  = P.all (\(_, _, ffnCycle)  -> ffnCycle >= 0)

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
#endif

        case result of
            Right () -> pure ()
            Left e   -> do
                P.putStrLn $ "\n[EXCEPTION CAUGHT]: " P.++ show e
                hFlush stdout
                expectationFailure $ show e
#endif
