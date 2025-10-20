{-# LANGUAGE CPP #-}
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
import Text.Printf (printf)
import LLaMa2.Types.LayerData ( Token, Temperature, Seed, ProcessingState (..) )

import LLaMa2.Types.ModelConfig ( VocabularySize )
import qualified LLaMa2.Top as Top ( topEntitySim, DecoderIntrospection (..) )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import LLaMa2.Numeric.Types (FixedPoint)
import Control.Monad (when)
import LLaMa2.Types.Parameters (DecoderParameters)

--------------------------------------------------------------------------------
-- Main entry point
--------------------------------------------------------------------------------

main :: IO ()
main = do
  Options {seed, tokenizerFile, modelFile, temperature, steps,
         prompt} <- OA.execParser $ OA.info (optionsParser OA.<**> OA.helper) OA.fullDesc
  modelFileContent <- BSL.readFile modelFile
  tokenizerFileContent <- BSL.readFile tokenizerFile
  runLLaMa2 modelFileContent tokenizerFileContent (realToFrac temperature) steps prompt seed

runLLaMa2 :: BSL.ByteString -> BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runLLaMa2 modelBinary tokenizerBinary temperature stepCount maybePrompt maybeSeed = do
  currentTime <- getPOSIXTime
  let
    randomSeed = fromIntegral $ fromMaybe (round currentTime) maybeSeed
    parseLLaMa2 = BG.runGet Parser.parseLLaMa2ConfigFile
    transformerConfig = parseLLaMa2 modelBinary
    tokenizer = T.buildTokenizer tokenizerBinary (C.natToNum @VocabularySize)

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
    generateTokensSimAutoregressive
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
    <*> OA.strOption (OA.long "tokenizer-file" <> OA.value
#if defined(MODEL_260K)
      "./data/tok512.bin"
#elif defined(MODEL_15M)
      "./data/tokenizer.bin"
#else
      "./data/tokenizer.bin"
#endif
    <> OA.help "Tokenizer binary file")
    <*> OA.strOption (OA.long "model-file" <> OA.value
#if defined(MODEL_260K)
      "./data/stories260K.bin" 
#elif defined(MODEL_15M)
      "./data/stories15M.bin" 
#else
      "./data/stories260K.bin" 
#endif
    <> OA.metavar "MODEL_FILE" <> OA.help "LLaMa2 binary file")
    <*> OA.option OA.auto (OA.long "temperature" <> OA.value 0.0 <> OA.metavar "TEMPERATURE" <> OA.help "Temperature")
    <*> OA.option OA.auto (OA.long "steps" <> OA.value 256 <> OA.metavar "STEPS" <> OA.help "Number of steps")
    <*> OA.optional (OA.strArgument (OA.metavar "PROMPT" <> OA.help "Initial prompt"))

printToken :: T.Tokenizer -> Token -> IO ()
printToken tokenizer tokenId = do
    BSC.putStr $ decodeToken tokenizer tokenId
    hFlush stdout

decodeToken :: T.Tokenizer -> Token -> BSC.ByteString
decodeToken tokenizer tokenId = T.decodePiece tokenizer (fromIntegral tokenId)

--------------------------------------------------------------------------------
-- Token Generation with Clash Simulation
--------------------------------------------------------------------------------

type DecoderInputState = (Token, [Token], Bool)

-- | Autoregressive token generation, one token at a time.
generateTokensSimAutoregressive
  :: DecoderParameters
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
    (firstToken, restPrompt) = case promptTokens of
      (t:ts) -> (t, ts)
      []     -> (1, [])

    advanceState (current, remPrompt, usingPrompt) (isReady, sampled)
      | not isReady = (current, remPrompt, usingPrompt)
      | otherwise   = case remPrompt of
                        (p:ps) -> (p, ps, True)
                        []     -> (sampled, [], False)

    temperature' = realToFrac temperature :: FixedPoint

    -- Scale simulation steps by 1000x
    simSteps = (stepCount + length promptTokens) * 1000

    inputSignals = C.fromList (DL.zip4 inputTokens inputValidFlags (repeat temperature') (repeat seed))

    (coreOutputsSignal, introspection) = bundledOutputs decoder inputSignals

    -- Sample core outputs
    coreOutputs :: [(Token, Bool)]
    coreOutputs = C.sampleN simSteps coreOutputsSignal

    -- Sample introspection fields separately
    statesSampled         = C.sampleN simSteps (Top.state introspection)
    layerIndicesSampled   = C.sampleN simSteps (Top.layerIndex introspection)
    readiesSampled        = C.sampleN simSteps (Top.ready introspection)
    attnDonesSampled      = C.sampleN simSteps (Top.attnDone introspection)
    ffnDonesSampled       = C.sampleN simSteps (Top.ffnDone introspection)

    -- Extract top-level outputs
    (outputTokens, readyFlags) = unzip coreOutputs

    -- Derive evolving state
    states :: [DecoderInputState]
    states = scanl advanceState (firstToken, restPrompt, True)
                (zip (drop 1 readyFlags) (drop 1 outputTokens))

    inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
    inputValidFlags = True       : [ usePrompt | (_, _, usePrompt) <- states ]

    sampledTokens = [ tok | (tok, isReady) <- zip outputTokens readyFlags, isReady ]

  -- Print header
  putStrLn "\nCycle | Layer | Stage | Ready | AttnDone | FFNDone | Token"
  putStrLn "-----------------------------------------------------------"

  -- Loop through sampled outputs and display selected signals
  let printCycle (cycleIdx, tok) = do
        let li     = fromIntegral (layerIndicesSampled !! cycleIdx) :: Int
            ps     = processingStage (statesSampled !! cycleIdx)
            rdy    = readiesSampled !! cycleIdx
            attn   = attnDonesSampled !! cycleIdx
            ffn    = ffnDonesSampled !! cycleIdx
            token  = coreOutputs !! cycleIdx
        when (cycleIdx `mod` 1000 == 0 || rdy || ffn) $
          putStrLn $
            printf "%5d | %5d | %-32s | %5s | %8s | %8s | %-8s | %-8s"
              cycleIdx
              li
              (show ps)
              (show rdy)
              (show attn)
              (show ffn)
              (show $ fst tok)
              (show $ decodeToken tokenizer (fst token))

  mapM_ printCycle (zip [0 :: Int ..] coreOutputs)

  mapM_ (printToken tokenizer) sampledTokens
  putStrLn ""
  pure (length sampledTokens)

bundledOutputs :: DecoderParameters
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> (C.Signal C.System (Token, Bool), Top.DecoderIntrospection C.System)
bundledOutputs decoder bundledInputs =
  (C.bundle (outToken, readyPulse), introspection)
 where
  (token, isValid, temperature, seed) = C.unbundle bundledInputs

  (outToken, readyPulse, introspection) =
    C.exposeClockResetEnable
      (Top.topEntitySim decoder token isValid temperature seed)
      CS.systemClockGen
      CS.resetGen
      CS.enableGen
