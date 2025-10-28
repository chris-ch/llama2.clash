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
import LLaMa2.Types.LayerData ( Token, Temperature, Seed, ProcessingState (..) )

import LLaMa2.Types.ModelConfig ( VocabularySize )
import qualified LLaMa2.Top as Top ( DecoderIntrospection (..), topEntityWithAxi )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import LLaMa2.Numeric.Types (FixedPoint)
import Control.Monad (when)
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified Simulation.DRAMBackedAxiSlave as DRAM

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
  let firstBytes = take 100 (BSL.unpack modelBinary)
  putStrLn $ "First 100 bytes of model file: " ++ show firstBytes
  putStrLn "✅ model loaded successfully"
  startTime <- getPOSIXTime

  countTokens <-
    generateTokensSimAutoregressive
      tokenizer
      modelBinary
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
  :: T.Tokenizer
  -> BSL.ByteString
  -> Int             -- ^ Number of tokens to generate
  -> [Token]         -- ^ Prompt tokens
  -> Float
  -> Seed
  -> IO Int
generateTokensSimAutoregressive tokenizer modelBinary stepCount promptTokens temperature seed = do
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

    (coreOutputsSignal, ddrMaster, ddrSlave, introspection) = bundledOutputs modelBinary inputSignals

    -- Sample DDR write signals
    ddrWValidSampled = C.sampleN simSteps (Master.wvalid ddrMaster)
    ddrWReadySampled = C.sampleN simSteps (Slave.wready ddrSlave)
    ddrBValidSampled = C.sampleN simSteps (Slave.bvalid ddrSlave)

    -- Sample core outputs
    coreOutputs :: [(Token, Bool)]
    coreOutputs = C.sampleN simSteps coreOutputsSignal

    -- Sample introspection fields separately
    statesSampled         = C.sampleN simSteps (Top.state introspection)
    layerIndicesSampled   = C.sampleN simSteps (Top.layerIndex introspection)
    readiesSampled        = C.sampleN simSteps (Top.ready introspection)
    --attnDonesSampled      = C.sampleN simSteps (Top.attnDone introspection)
    ffnDonesSampled       = C.sampleN simSteps (Top.ffnDone introspection)
    weightValidSampled = C.sampleN simSteps (Top.weightStreamValid introspection)
    weightSampleSampled = C.sampleN simSteps (Top.parsedWeightSample introspection)
    layerChangeSampled = C.sampleN simSteps (Top.layerChangeDetected introspection)
    sysStateSampled = C.sampleN simSteps (Top.sysState introspection)

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
  putStrLn "\nCycle | Layer | Stage              | Boot  | Tok Rdy  | FFNDone  | WgtValid  | LayerChg  | WgtSample | TokID   |   Tok   | SsyState | ddrWValid | ddrWReady| ddrBValid"
  putStrLn "-------------------------------------------------------------------------------------------------------------------------------------------------------------------"

  -- Loop through sampled outputs and display selected signals
  let printCycle (cycleIdx, token') = do
        let 
          li     = fromIntegral (layerIndicesSampled !! cycleIdx) :: Int
          ps     = processingStage (statesSampled !! cycleIdx)
          rdy    = readiesSampled !! cycleIdx
          ffn    = ffnDonesSampled !! cycleIdx
          wValid = weightValidSampled !! cycleIdx
          layChg = layerChangeSampled !! cycleIdx
          wSample = weightSampleSampled !! cycleIdx
          token  = coreOutputs !! cycleIdx
          sysSt  = sysStateSampled !! cycleIdx
          ddrWValid = ddrWValidSampled !! cycleIdx
          ddrWReady = ddrWReadySampled !! cycleIdx
          ddrBValid = ddrBValidSampled !! cycleIdx
        when (cycleIdx `mod` 1000 == 0 || rdy || ffn) $
          putStrLn $
            printf "%5d | %5d | %-18s | %5s | %8s | %8s | %9s |  %9d | %8s | %8s | %8s | %8s | %8s | %8s"
              cycleIdx
              li
              (show ps)
              (show rdy)
              (show ffn)
              (show wValid)
              (show layChg)
              wSample
              (show $ fst token)
              (show $ decodeToken tokenizer (fst token))
              (show sysSt)
              (show ddrWValid)
              (show ddrWReady)
              (show ddrBValid)

  mapM_ printCycle (zip [0 :: Int ..] coreOutputs)

  mapM_ (printToken tokenizer) sampledTokens
  putStrLn ""
  pure (length sampledTokens)

-- | Boot and DDR simulation overview
-- 
-- During boot, the system copies model weights from the eMMC (AXI slave)
-- into DDR (AXI slave) using the weight loader. Once boot completes, all
-- parameters reside in DDR, and the decoder streams weights directly from
-- there for inference.
--
-- In simulation, both eMMC and DDR are modeled using 'FileAxi.createFileBackedAxiSlave',
-- which serves data directly from the same model binary. This means no actual
-- data movement occurs; the eMMC→DDR copy is functionally simulated through AXI
-- transactions backed by an in-memory ByteString.
--
-- The 'bypassBoot' signal skips the boot sequence entirely, causing the system to
-- assume DDR is already loaded and ready. Useful for fast functional simulation.
bundledOutputs
  :: BSL.ByteString
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> ( C.Signal C.System (Token, Bool)
     , Master.AxiMasterOut C.System
     , Slave.AxiSlaveIn C.System
     , Top.DecoderIntrospection C.System
     )
bundledOutputs modelBinary bundledInputs =
  (C.bundle (tokenOut, validOut), ddrMaster, ddrSlave, introspection)
 where
  (token, isValid, temperature, seed) = C.unbundle bundledInputs

  powerOn = C.pure True

  -- For DDR: also serve from file (simulating that boot already copied data there)
  ddrSlave = C.exposeClockResetEnable
    --(FileAxi.createFileBackedAxiSlave modelBinary ddrMaster)
    (DRAM.createDRAMBackedAxiSlave modelBinary ddrMaster)
    C.systemClockGen
    C.resetGen
    C.enableGen

  -- Create the feedback loop: masters drive slaves, slaves feed back to decoder
  (tokenOut, validOut, ddrMaster, introspection) =
    C.exposeClockResetEnable
      (Top.topEntityWithAxi ddrSlave powerOn token isValid temperature seed)
      C.systemClockGen
      C.resetGen
      C.enableGen

