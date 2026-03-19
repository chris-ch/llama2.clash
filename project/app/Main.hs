module Main (main) where

import Prelude

import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Options.Applicative as OA
import qualified Clash.Prelude as C

import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import System.IO (hFlush, stdout)
import Text.Printf (printf)
import LLaMa2.Types.LayerData ( Token, Seed )
import LLaMa2.Numeric.Types (FixedPoint)

import LLaMa2.Types.ModelConfig ( VocabularySize, ModelDimension, NumQueryHeads, NumKeyValueHeads, NumLayers, HiddenDimension, SequenceLength, RotaryPositionalEmbeddingDimension )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import Control.Monad (when)
import qualified Simulation.ParamsPlaceholder as PARAM (decoderConst, tokenizerFile)
import qualified LLaMa2.Decoder.Decoder as Decoder
import Simulation.Parameters (DecoderParameters)
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import qualified SimUtils

--------------------------------------------------------------------------------
-- Main entry point
--------------------------------------------------------------------------------

main :: IO ()
main = do
  Options {seed, tokenizerFile, temperature, steps,
         prompt} <- OA.execParser $ OA.info (optionsParser OA.<**> OA.helper) OA.fullDesc
  tokenizerFileContent <- BSL.readFile tokenizerFile
  runLLaMa2 tokenizerFileContent (realToFrac temperature) steps prompt seed

runLLaMa2 :: BSL.ByteString -> Float -> Int -> Maybe String -> Maybe Int -> IO ()
runLLaMa2 tokenizerBinary temperature stepCount maybePrompt maybeSeed = do
  currentTime <- getPOSIXTime
  let
    randomSeed = fromIntegral $ fromMaybe (round currentTime) maybeSeed
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
  startTime <- getPOSIXTime

  countTokens <-
    generateTokensSimAutoregressive
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
    temperature :: Double,
    steps :: Int,
    prompt :: Maybe String
  }

-- Parser for command-line options
optionsParser :: OA.Parser Options
optionsParser =
  Options
    <$> OA.optional (OA.option OA.auto (OA.long "seed" <> OA.help "Seed for debugging"))
    <*> OA.strOption (OA.long "tokenizer-file" <> OA.value PARAM.tokenizerFile
    <> OA.help "Tokenizer binary file")
    <*> OA.option OA.auto (OA.long "temperature" <> OA.value 0.0 <> OA.metavar "TEMPERATURE" <> OA.help "Temperature")
    <*> OA.option OA.auto (OA.long "steps" <> OA.value 256 <> OA.metavar "STEPS" <> OA.help "Number of steps")
    <*> OA.optional (OA.strArgument (OA.metavar "PROMPT" <> OA.help "Initial prompt"))

printToken :: T.Tokenizer -> Token -> IO ()
printToken tokenizer tokenId = do
    BSC.putStrLn $ decodeToken tokenizer tokenId
    hFlush stdout

decodeToken :: T.Tokenizer -> Token -> BSC.ByteString
decodeToken tokenizer tokenId = T.decodePiece tokenizer (fromIntegral tokenId)

--------------------------------------------------------------------------------
-- Token Generation with Clash Simulation
--------------------------------------------------------------------------------

type DecoderInputState = (Token, [Token], Bool)

-- | Autoregressive token generation, one token at a time.
generateTokensSimAutoregressive
  :: T.Tokenizer
  -> Int             -- ^ Number of tokens to generate
  -> [Token]         -- ^ Prompt tokens
  -> Float
  -> Seed
  -> IO Int
generateTokensSimAutoregressive tokenizer stepCount promptTokens temperature seed = do
  putStrLn $ "Vocabulary size: " ++ show (C.natToNum @VocabularySize :: Int)
  putStrLn $ "Model dimension: " ++ show (C.natToNum @ModelDimension :: Int)
  putStrLn $ "Hidden dimension: " ++ show (C.natToNum @HiddenDimension :: Int)
  putStrLn $ "Rotary Positional Embedding dimension: " ++ show (C.natToNum @RotaryPositionalEmbeddingDimension :: Int)
  putStrLn $ "Sequence length: " ++ show (C.natToNum @SequenceLength :: Int)
  putStrLn $ "# Query heads: " ++ show (C.natToNum @NumQueryHeads :: Int)
  putStrLn $ "# Key-Value heads: " ++ show (C.natToNum @NumKeyValueHeads :: Int)
  putStrLn $ "# Layers: " ++ show (C.natToNum @NumLayers :: Int)
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

    -- Scale simulation steps by 1000x
    -- Each token takes ~38000 cycles (5 layers × ~7000 cycles/layer through DRAM arbiter).
    -- Use 50000x multiplier so (stepCount + promptLen) * 50000 covers all tokens.
    -- Note: run with --steps 1 for quick testing (~100k cycles = ~50 min).
    simSteps = (stepCount + length promptTokens) * 50000

    temperature' = realToFrac temperature :: FixedPoint

    inputSignals = C.fromList (DL.zip inputTokens inputValidFlags)

    (coreOutputsSignal, introspection) = bundledOutputs temperature' seed inputSignals

    -- Sample core outputs
    coreOutputs :: [(Token, Bool)]
    coreOutputs = C.sampleN simSteps coreOutputsSignal

    -- Sample introspection fields
    layerIndicesSampled  = C.sampleN simSteps (Decoder.layerIndex introspection)
    readiesSampled       = C.sampleN simSteps (Decoder.ready introspection)
    layerValidInsSampled = C.sampleN simSteps (Decoder.layerValidIn introspection)
    layerDonesSampled    = C.sampleN simSteps (Decoder.layerDone introspection)
    cycleCountSampled    = C.sampleN simSteps (Decoder.cycleCount introspection)
    ffnOut0Sampled       = C.sampleN simSteps (Decoder.ffnOut0 introspection)

    -- Extract top-level outputs
    (outputTokens, readyFlags) = unzip coreOutputs

    -- Derive evolving state
    states :: [DecoderInputState]
    states = scanl advanceState (firstToken, restPrompt, True)
                (zip (drop 1 readyFlags) (drop 1 outputTokens))

    inputTokens     = firstToken : [ tok | (tok, _, _) <- states ]
    inputValidFlags = True       : [ usePrompt | (_, _, usePrompt) <- states ]

    sampledTokens = [ tok | (tok, isReady) <- zip outputTokens readyFlags, isReady ]

  putStrLn $ "Simulating " ++ show simSteps ++ " cycles..."
  putStrLn "This may take a moment..."

  -- Print header
  putStrLn "\nCycle | Layer | Tok Rdy | LayerDone |     Tok     | LayerValid"
  putStrLn "---------------------------------------------------------------"

  let printCycle ioIdx = do
        let hwCycle = fromIntegral $ cycleCountSampled !! ioIdx
        let
          li          = fromIntegral (layerIndicesSampled !! hwCycle) :: Int
          rdy         = readiesSampled !! hwCycle
          done        = layerDonesSampled !! hwCycle
          token       = coreOutputs !! hwCycle
          layerValidIn = layerValidInsSampled !! hwCycle

        let ffnOut0 = ffnOut0Sampled !! hwCycle
        when (hwCycle `mod` 10000 == 0 || rdy || done || layerValidIn) $
          putStrLn $
            printf "%5d | %5d | %7s | %8s | %11s | %10s | ffnOut0=%s"
              hwCycle
              li
              (show rdy)
              (show done)
              (show $ decodeToken tokenizer (fst token))
              (show layerValidIn)
              (if done then show (realToFrac ffnOut0 :: Double) else "-")

  mapM_ printCycle [0 :: Int ..]

  mapM_ (printToken tokenizer) sampledTokens
  putStrLn ""
  pure (length sampledTokens)

-- | DDR simulation overview
bundledOutputs
  :: FixedPoint
  -> Seed
  -> C.Signal C.System (Token, Bool)
  -> ( C.Signal C.System (Token, Bool)
     , Decoder.DecoderIntrospection C.System
     )
bundledOutputs temperature seed bundledInputs =
  (C.bundle (tokenOut, validOut), introspection)
 where
  (token, isValid) = C.unbundle bundledInputs

  params :: DecoderParameters
  params = PARAM.decoderConst

  -- Instantiate cycle counter with explicit clock/reset/enable
  cycleCounter :: C.Signal C.System (C.Unsigned 32)
  cycleCounter =
    C.exposeClockResetEnable
      SimUtils.makeCycleCounter
      C.systemClockGen
      C.resetGen
      C.enableGen

  -- Create weights DDR simulator
  dramSlaveIn = C.exposeClockResetEnable
    (DRAMSlave.createDRAMBackedAxiSlave cycleCounter params ddrMaster)
    C.systemClockGen
    C.resetGen
    C.enableGen

  -- KV cache DRAM slaves (lazy circular dependency with decoder's KV masters)
  kvDramSlaves = C.exposeClockResetEnable
    (C.map (DRAMSlave.createKVCacheDRAMSlave cycleCounter) kvMasters)
    C.systemClockGen
    C.resetGen
    C.enableGen

  -- Decoder with AXI feedback loop
  (ddrMaster, kvMasters, tokenOut, validOut, introspection) =
    C.exposeClockResetEnable
      (Decoder.decoder cycleCounter dramSlaveIn kvDramSlaves token isValid (pure False) (pure temperature) (pure seed))
      C.systemClockGen
      C.resetGen
      C.enableGen
