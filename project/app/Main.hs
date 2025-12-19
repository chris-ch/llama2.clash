{-# LANGUAGE CPP #-}
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
import LLaMa2.Types.LayerData ( Token, Temperature, Seed, LayerData (..) )

import LLaMa2.Types.ModelConfig ( VocabularySize, ModelDimension, NumQueryHeads, NumKeyValueHeads, NumLayers, HiddenDimension, SequenceLength, RotaryPositionalEmbeddingDimension )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import LLaMa2.Numeric.Types (FixedPoint)
import Control.Monad (when)
import qualified Simulation.ParamsPlaceholder as PARAM (decoderConst)
import qualified LLaMa2.Decoder.Decoder as Decoder
import Simulation.Parameters (DecoderParameters)
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import Numeric (showHex)

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
    <*> OA.strOption (OA.long "tokenizer-file" <> OA.value
#if defined(MODEL_260K)
      "./data/tok512.bin"
#elif defined(MODEL_15M)
      "./data/tokenizer.bin"
#else
      "./data/tokenizer.bin"
#endif
    <> OA.help "Tokenizer binary file")
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

fromSFixed :: C.SFixed 12 20 -> Float
fromSFixed x = realToFrac (toRational x)

-- | Convert a Clash vector of signed fixed-point numbers to a list of Float.
vecToFloatList :: C.Vec n (C.SFixed 12 20) -> [Float]
vecToFloatList = map fromSFixed . C.toList
--   fromSFixed :: SFixed a b -> Rational
--   Rational → Float via implicit conversion

-- | Euclidean (L2) norm of a fixed-point vector, returned as Float.
normVec :: C.Vec n (C.SFixed 12 20) -> Float
normVec v = sqrt (sum squares)
          where
            floats = vecToFloatList v
            squares = map (\x -> x * x) floats

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

    temperature' = realToFrac temperature :: FixedPoint

    -- Scale simulation steps by 1000x
    simSteps = (stepCount + length promptTokens) * 1000
    --simSteps = 250_000  -- Just 250K cycles to test boot completes

    inputSignals = C.fromList (DL.zip4 inputTokens inputValidFlags (repeat temperature') (repeat seed))

    (coreOutputsSignal, introspection) = bundledOutputs inputSignals

    -- Sample core outputs
    coreOutputs :: [(Token, Bool)]
    coreOutputs = C.sampleN simSteps coreOutputsSignal

    -- Sample introspection fields separately
    layerIndicesSampled   = C.sampleN simSteps (Decoder.layerIndex introspection)
    readiesSampled        = C.sampleN simSteps (Decoder.ready introspection)
    layerValidInsSampled        = C.sampleN simSteps (Decoder.layerValidIn introspection)
    qkvDonesSampled      = C.sampleN simSteps (Decoder.qkvDone introspection)
    attnDonesSampled      = C.sampleN simSteps (Decoder.attnDone introspection)
    ffnDonesSampled       = C.sampleN simSteps (Decoder.ffnDone introspection)
    layerChangeSampled = C.sampleN simSteps (Decoder.layerChangeDetected introspection)
    layerOutputSampled =  C.sampleN simSteps (Decoder.layerOutput introspection)
    layerDataSampled :: [LayerData]
    layerDataSampled =  C.sampleN simSteps (Decoder.layerData introspection)
    loadTriggerActiveSampled = C.sampleN simSteps (Decoder.loadTriggerActive introspection)
    paramQ0Row0Sampled = C.sampleN simSteps (Decoder.paramQ0Row0 introspection)

    dbgRowIndexSampled  = C.sampleN simSteps (Decoder.dbgRowIndex introspection)
    dbgStateSampled     = C.sampleN simSteps (Decoder.dbgState introspection)
    dbgFirstMantSampled = C.sampleN simSteps (Decoder.dbgFirstMant introspection)
    dbgRowResultSampled = C.sampleN simSteps (Decoder.dbgRowResult introspection)
    dbgRowDoneSampled   = C.sampleN simSteps (Decoder.dbgRowDone introspection)
    dbgFetchValidSampled= C.sampleN simSteps (Decoder.dbgFetchValid introspection)
    dbgFetchedWordSampled = C.sampleN simSteps (Decoder.dbgFetchedWord introspection)
    seqPosSampled = C.sampleN simSteps (Decoder.seqPos introspection)

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
  putStrLn "\nCycle | Layer | Tok Rdy | QKVDone | AttnDone | FFNDone | WgtValid | norm(attn) | norm(out) |     Tok     | LayerValid | loadTriggerActive | paramQ0Row0 |  dbgRowIdx  |     dbgState    | dbgFetchValid"
  putStrLn "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

  -- Loop through sampled outputs and display selected signals
  let printCycle (cycleIdx, token') = do
        let

          attnOut :: C.Vec ModelDimension FixedPoint
          attnOut = attentionOutput (layerDataSampled !! cycleIdx)

          li     = fromIntegral (layerIndicesSampled !! cycleIdx) :: Int
          rdy    = readiesSampled !! cycleIdx
          qkv    = qkvDonesSampled !! cycleIdx
          attn    = attnDonesSampled !! cycleIdx
          ffn    = ffnDonesSampled !! cycleIdx
          layChg = layerChangeSampled !! cycleIdx
          layerOutputNorm = normVec $ layerOutputSampled !! cycleIdx
          attnOutNorm = normVec attnOut
          token  = coreOutputs !! cycleIdx
          layerValidIn = layerValidInsSampled !! cycleIdx
          loadTriggerActive = loadTriggerActiveSampled !! cycleIdx
          paramQ0Row0 = paramQ0Row0Sampled !! cycleIdx

          dbgRowIdx      = dbgRowIndexSampled !! cycleIdx
          dbgSt          = dbgStateSampled !! cycleIdx
          dbgFirstMant   = dbgFirstMantSampled !! cycleIdx
          dbgRowRes      = dbgRowResultSampled !! cycleIdx
          dbgRowDone     = dbgRowDoneSampled !! cycleIdx
          dbgFetchValid  = dbgFetchValidSampled !! cycleIdx
          dbgFetchedWord  = dbgFetchedWordSampled !! cycleIdx
          seqPosition  = seqPosSampled !! cycleIdx

        when (cycleIdx `mod` 10000 == 0 || rdy || qkv || attn || ffn || layChg || layerValidIn || loadTriggerActive || dbgRowDone) $
          putStrLn $
            printf "%5d | %5d | %7s | %7s | %8s | %8s | %8s | %10.4f | %9.4f | %11s | %10s | %15s | %14s| %10s | %16s"
              cycleIdx
              li
              (show rdy)
              (show qkv)
              (show attn)
              (show ffn)
              (show layChg)
              attnOutNorm
              layerOutputNorm
              (show $ decodeToken tokenizer (fst token))
              (show layerValidIn)
              (show loadTriggerActive)
              --(if wValid then showHex rawWeightStream "" else "N/A")
              (show paramQ0Row0)
              (show dbgRowIdx)
              (show dbgSt)

  mapM_ printCycle (zip [0 :: Int ..] coreOutputs)

  mapM_ (printToken tokenizer) sampledTokens
  putStrLn ""
  pure (length sampledTokens)

bv512ToHex :: C.BitVector 512 -> String
bv512ToHex bv = case C.maybeIsX bv of
    Just _  -> "<undefined>"
    Nothing -> "0x" ++ padTo128 '0' (showHex bv "")
  where
    padTo128 c s = replicate (128 - length s) c ++ s
    
-- | DDR simulation overview
bundledOutputs
  :: C.Signal C.System (Token, Bool, Temperature, Seed)
  -> ( C.Signal C.System (Token, Bool)
     , Decoder.DecoderIntrospection C.System
     )
bundledOutputs bundledInputs =
  (C.bundle (tokenOut, validOut), introspection)
 where
  (token, isValid, temperature, seed) = C.unbundle bundledInputs

  params :: DecoderParameters
  params = PARAM.decoderConst

  -- Create DDR simulator
  dramSlaveIn = C.exposeClockResetEnable
    (DRAMSlave.createDRAMBackedAxiSlave params ddrMaster)
    C.systemClockGen
    C.resetGen
    C.enableGen

  -- Decoder with AXI feedback loop
  (ddrMaster, tokenOut, validOut, introspection) =
    C.exposeClockResetEnable
      (Decoder.decoder dramSlaveIn params token isValid temperature seed)
      C.systemClockGen
      C.resetGen
      C.enableGen

