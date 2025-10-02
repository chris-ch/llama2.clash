module Main (main) where

import Prelude

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Options.Applicative as OA
import qualified Clash.Prelude as C
import qualified Clash.Signal as CS
import qualified Parser

import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import System.IO (hFlush, stdout)
import Control.Monad (unless)
import Text.Printf (printf)
import Model.Core.Types ( Token, Temperature, Seed )

import Model.Config ( VocabularySize, NumLayers )
import qualified Model.Top as Top ( topEntity )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import Model.Layers.TransformerLayer (TransformerDecoderComponent (..))
import Model.Numeric.Types (FixedPoint)
import qualified Model.TopDebug as TopDebug


--------------------------------------------------------------------------------
-- Main entry point
--------------------------------------------------------------------------------

main :: IO ()
main = do
  Options {seed, tokenizerFile, modelFile, temperature, steps,
         prompt} <- OA.execParser $ OA.info (optionsParser OA.<**> OA.helper) OA.fullDesc
  modelFileContent <- BSL.readFile modelFile
  tokenizerFileContent <- BSL.readFile tokenizerFile
  runModel modelFileContent tokenizerFileContent (realToFrac temperature) steps prompt seed

runModel :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runModel modelBinary tokenizerBinary temperature stepCount maybePrompt maybeSeed = do
  currentTime <- getPOSIXTime
  let
    randomSeed = fromIntegral $ fromMaybe (round currentTime) maybeSeed
    parseModel = BG.runGet Parser.parseModelConfigFile
    transformerConfig = parseModel modelBinary
    tokenizer = T.buildTokenizer tokenizerBinary (C.natToNum @VocabularySize)

  -- Handle prompt tokenization more carefully
  initialTokens <- case maybePrompt of
    Nothing -> do
      putStrLn "No prompt provided, starting with BOS token (1)"
      pure [1]
    Just promptText -> do
      let encodedTokens = T.encodeTokens tokenizer (BSC.pack promptText)
          tokenIds = map fromIntegral encodedTokens :: [Token]
      case tokenIds of
        [] -> do
          -- Empty tokenization, ensure we have BOS
          putStrLn "Empty tokenization, adding BOS token (1)"
          return [1]
        (1:_) -> do
          -- Already starts with BOS, good
          putStrLn $ "Prompt already starts with BOS: " ++ show tokenIds
          return tokenIds
        _ -> do
          -- Doesn't start with BOS, prepend it
          putStrLn $ "Prepending BOS to prompt tokens: " ++ show (1 : tokenIds)
          return (1 : tokenIds)

  putStrLn "✅ model loaded successfully"
  startTime <- getPOSIXTime

  countTokens <-
    generateTokensSimAutoregressiveDebug
      transformerConfig
      tokenizer
      stepCount
      initialTokens
      temperature
      randomSeed

  endTime <- getPOSIXTime
  let elapsedSeconds :: Integer
      elapsedSeconds = round (endTime - startTime)
      throughputTokensPerSec :: Float
      throughputTokensPerSec = fromIntegral countTokens / fromIntegral elapsedSeconds
  printf "\nduration: %ds - (%.02f tokens/s)\n" elapsedSeconds throughputTokensPerSec
  return ()

--------------------------------------------------------------------------------
-- Options
--------------------------------------------------------------------------------

data Options = Options
  { seed :: Maybe Int,
    tokenizerFile :: FilePath,
    modelFile :: FilePath,
    temperature :: Double,
    steps :: Int,
    prompt :: Maybe String
  }

-- Parser for command-line options
optionsParser :: OA.Parser Options
optionsParser =
  Options
    <$> OA.optional (OA.option OA.auto (OA.long "seed" <> OA.help "Seed for debugging"))
    <*> OA.strOption (OA.long "tokenizer-file" <> OA.value "./data/tokenizer.bin" <> OA.help "Tokenizer binary file")
    <*> OA.strOption (OA.long "model-file" <> OA.value "./data/stories110M.bin" <> OA.metavar "MODEL_FILE" <> OA.help "Model binary file")
    <*> OA.option OA.auto (OA.long "temperature" <> OA.value 0.0 <> OA.metavar "TEMPERATURE" <> OA.help "Temperature")
    <*> OA.option OA.auto (OA.long "steps" <> OA.value 256 <> OA.metavar "STEPS" <> OA.help "Number of steps")
    <*> OA.optional (OA.strArgument (OA.metavar "PROMPT" <> OA.help "Initial prompt"))

printToken :: T.Tokenizer -> Token -> IO ()
printToken tokenizer tokenId = do
    BSC.putStr (T.decodePiece tokenizer (fromIntegral tokenId) (fromIntegral tokenId))
    hFlush stdout

--------------------------------------------------------------------------------
-- Token Generation with Clash Simulation
--------------------------------------------------------------------------------

type DecoderInputState = (Token, [Token], Bool)

-- | Autoregressive token generation, one token at a time.
generateTokensSimAutoregressive
  :: TransformerDecoderComponent
  -> T.Tokenizer
  -> Int             -- ^ Number of tokens to generate
  -> [Token]         -- ^ Prompt tokens
  -> Float
  -> Seed
  -> IO Int
generateTokensSimAutoregressive decoder tokenizer stepCount promptTokens temperature seed = do
  putStrLn $ "✅ Prompt: " ++ show promptTokens
  hFlush stdout

  let
    -- split prompt into first token + rest (BOS fallback)
    (firstToken, restPrompt) =
      case promptTokens of
        (t:ts) -> (t, ts)
        []     -> (1, [])

    -- state: (currentInputToken, remainingPrompt, stillUsingPrompt)
    advanceState :: DecoderInputState -> (Bool, Token) -> DecoderInputState
    advanceState (current, remPrompt, usingPrompt) (isReady, sampled)
      | not isReady = (current, remPrompt, usingPrompt)
      | otherwise   = case remPrompt of
                        (p:ps) -> (p, ps, True)
                        []     -> (sampled, [], False)

    temperature' = realToFrac temperature :: FixedPoint

    outputs :: [(Token, Bool)]
    outputs =
      CS.simulate (bundledOutputs decoder)
                  (DL.zip4 inputTokens inputValidFlags (repeat temperature') (repeat seed))

    (outputTokens, readyFlags) = unzip outputs

    -- build the evolution of the input-state using the future ready flags & outputs
    states :: [DecoderInputState]
    states = scanl advanceState (firstToken, restPrompt, True)
                (zip (drop 1 readyFlags) (drop 1 outputTokens))

    -- inputs fed to the model (matches original structure)
    inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
    inputValidFlags = True       : [ usePrompt | (_, _, usePrompt) <- states ]

    -- collect sampled tokens (only where isReady == True)
    sampledTokens = [ tok | (tok, isReady) <- outputs, isReady ]

    promptLength    = length promptTokens
    generatedTokens = take stepCount (drop promptLength sampledTokens)
    emittedTokens   = promptTokens ++ generatedTokens
    limitedEmitted  = take (promptLength + stepCount) emittedTokens

  mapM_ (printToken tokenizer) limitedEmitted
  putStrLn ""
  return stepCount

bundledOutputs :: TransformerDecoderComponent
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> C.Signal C.System (Token, Bool)
bundledOutputs decoder bundledInputs =
  C.bundle $
    C.exposeClockResetEnable (Top.topEntity decoder token isValid temperature seed)
                             CS.systemClockGen
                             CS.resetGen
                             CS.enableGen
  where
    (token, isValid, temperature, seed) = C.unbundle bundledInputs

-- ==================================
-- DEBUGGING
-- ==================================

bundledOutputsDebug
  :: TransformerDecoderComponent
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> C.Signal C.System (Token, Bool, C.Vec NumLayers Bool, C.Vec NumLayers Bool)
bundledOutputsDebug decoder bundledInputs =
  C.bundle $
    C.exposeClockResetEnable
      (TopDebug.topEntityDebug decoder token isValid temperature seed)
      CS.systemClockGen
      CS.resetGen
      CS.enableGen
 where
  (token, isValid, temperature, seed) = C.unbundle bundledInputs

-- A debug generation routine that prints row errors when they occur.
-- | Print at most the cycles needed to emit (promptLength + stepCount) tokens.
--   This avoids consuming the infinite stream from 'simulate'.
-- | Debug generator that:
--   - prints the prompt immediately,
--   - then streams tokens as 'ready' pulses arrive,
--   - prints row error diagnostics per-cycle on-the-fly, and
--   - never traverses unbounded lists looking for the Nth 'True'.
--   It also has a large but finite cycle cap to avoid true hangs if 'ready' never fires.
generateTokensSimAutoregressiveDebug
  :: TransformerDecoderComponent
  -> T.Tokenizer
  -> Int
  -> [Token]
  -> Float
  -> Seed
  -> IO Int
generateTokensSimAutoregressiveDebug decoder tokenizer stepCount promptTokens temperature seed = do
  putStrLn $ "✅ Prompt: " ++ show promptTokens
  hFlush stdout
  let
    (firstToken, restPrompt) =
      case promptTokens of { (t:ts) -> (t, ts); [] -> (1, []) }

    -- Same state machine as your non-debug path; advances on ready pulses.
    advanceState :: (Token,[Token],Bool) -> (Bool,Token) -> (Token,[Token],Bool)
    advanceState (current, remPrompt, usingPrompt) (isReady, sampled)
      | not isReady = (current, remPrompt, usingPrompt)
      | otherwise   = case remPrompt of
                        (p:ps) -> (p, ps, True)
                        []     -> (sampled, [], False)

    temperature' = realToFrac temperature :: FixedPoint

    -- Classic knot-tying: inputs depend on future outputs via 'states'.
    outputsDbg :: [(Token, Bool, C.Vec NumLayers Bool, C.Vec NumLayers Bool)]
    outputsDbg =
      CS.simulate (bundledOutputsDebug decoder)
                  (DL.zip4 inputTokens inputValidFlags (repeat temperature') (repeat seed))

    (outputTokens, readyFlags, _kErrVecs, _vErrVecs) = DL.unzip4 outputsDbg

    states :: [(Token,[Token],Bool)]
    states = scanl advanceState (firstToken, restPrompt, True)
                (zip (drop 1 readyFlags) (drop 1 outputTokens))

    inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
    inputValidFlags = True       : [ use | (_, _, use) <- states ]

    promptLength = length promptTokens

    -- Pretty-print one cycle's row errors if any.
    printErr :: (Int, (C.Vec NumLayers Bool, C.Vec NumLayers Bool)) -> IO ()
    printErr (cyc,(kE,vE)) =
      let ks  = zip [(0 :: Int)..] (C.toList kE)
          vs  = zip [(0 :: Int)..] (C.toList vE)
          badK = [ i | (i,True) <- ks ]
          badV = [ i | (i,True) <- vs ]
      in unless (null badK && null badV) $
           putStrLn $ "⚠️  RowErr at cycle " ++ show cyc
                    ++ " K(layers)=" ++ show badK
                    ++ " V(layers)=" ++ show badV

    -- Cycle budget to avoid true hangs if 'ready' never arrives.
    maxCycles :: Int
    maxCycles = 1000000

    -- Stream over the simulated outputs one element at a time.
    -- We print prompt first, then print 'stepCount' generated tokens as they arrive.
    loop !cyc !readySeen !gen ((tok,isReady,kE,vE):rest) = do
      -- per-cycle diagnostics
      printErr (cyc,(kE,vE))

      -- optional: stop after a very large number of cycles to avoid infinite wait
      if cyc >= maxCycles then do
        putStrLn $ "\n⏱️  Gave up after " ++ show cyc ++ " cycles without finishing."
                 ++ " readySeen=" ++ show readySeen ++ " generated=" ++ show gen
        putStrLn ""
        pure gen
      else if not isReady then
        loop (cyc+1) readySeen gen rest
      else
        -- A 'ready' pulse observed this cycle.
        if readySeen < promptLength then
          -- Warm-up 'ready's for the prompt (we already printed the prompt explicitly).
          loop (cyc+1) (readySeen+1) gen rest
        else if gen < stepCount then do
          -- Print a sampled token
          printToken tokenizer tok
          loop (cyc+1) (readySeen+1) (gen+1) rest
        else do
          -- Done: printed all requested tokens
          putStrLn ""
          pure gen

    loop _ _ gen [] = do
      -- 'simulate' never ends, so this clause should be unreachable.
      putStrLn $ "\nUnexpected end of simulation stream; generated=" ++ show gen
      pure gen

  -- Print prompt immediately so users see activity right away.
  mapM_ (printToken tokenizer) promptTokens
  hFlush stdout

  -- Walk the simulation stream until we print 'stepCount' generated tokens (or hit the cycle cap).
  loop 0 0 0 outputsDbg