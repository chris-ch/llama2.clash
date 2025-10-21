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
import qualified LLaMa2.Top as Top ( DecoderIntrospection (..), topEntityWithAxi )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import LLaMa2.Numeric.Types (FixedPoint)
import Control.Monad (when)
import LLaMa2.Types.Parameters (DecoderParameters)
import LLaMa2.Memory.AXI (AxiSlaveIn, AxiMasterOut (..))
import LLaMa2.Memory.FakeAxiSlave (createFakeAxiSlave)
import LLaMa2.Memory.SimAxiSlave (createSimAxiSlave)
import qualified LLaMa2.Memory.FileBackedAxiSlave as FileAxi
import qualified LLaMa2.Memory.SimAxiSlave as SimAxi
import LLaMa2.Memory.FileBackedAxiSlave (ReadState)

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
  let firstBytes = take 100 (BSL.unpack modelBinary)
  putStrLn $ "First 100 bytes of model file: " ++ show firstBytes
  putStrLn "✅ model loaded successfully"
  startTime <- getPOSIXTime

  countTokens <-
    generateTokensSimAutoregressive
      transformerConfig
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
  :: DecoderParameters
  -> T.Tokenizer
  -> BSL.ByteString
  -> Int             -- ^ Number of tokens to generate
  -> [Token]         -- ^ Prompt tokens
  -> Float
  -> Seed
  -> IO Int
generateTokensSimAutoregressive decoder tokenizer modelBinary stepCount promptTokens temperature seed = do
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
    --simSteps = 18000  -- Just 25K cycles to test boot completes

    inputSignals = C.fromList (DL.zip4 inputTokens inputValidFlags (repeat temperature') (repeat seed))

    (coreOutputsSignal, emmcMaster, ddrMaster, weightsReady, bootProgress,
     introspection, emmcSlaveArHandshakeCount, ddrSlaveArHandshakeCount
     ) = bundledOutputs decoder modelBinary inputSignals

    emmcMasterRReadySampled = C.sampleN simSteps (LLaMa2.Memory.AXI.rready emmcMaster)
    
    -- Sample core outputs
    coreOutputs :: [(Token, Bool)]
    coreOutputs = C.sampleN simSteps coreOutputsSignal

    -- Sample introspection fields separately
    statesSampled         = C.sampleN simSteps (Top.state introspection)
    layerIndicesSampled   = C.sampleN simSteps (Top.layerIndex introspection)
    readiesSampled        = C.sampleN simSteps (Top.ready introspection)
    attnDonesSampled      = C.sampleN simSteps (Top.attnDone introspection)
    ffnDonesSampled       = C.sampleN simSteps (Top.ffnDone introspection)
    weightsReadySampled = C.sampleN simSteps weightsReady
    weightValidSampled = C.sampleN simSteps (Top.weightStreamValid introspection)
    weightSampleSampled = C.sampleN simSteps (Top.parsedWeightSample introspection)
    bootProgressSampled = C.sampleN simSteps (Top.bootProgressBytes introspection)
    layerChangeSampled = C.sampleN simSteps (Top.layerChangeDetected introspection)
    sysStateSampled = C.sampleN simSteps (Top.sysState introspection)
    bootStateSampled = C.sampleN simSteps (Top.bootState introspection)
    emmcHandshakesSampled = C.sampleN simSteps emmcSlaveArHandshakeCount
    ddrHandshakesSampled = C.sampleN simSteps ddrSlaveArHandshakeCount

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
  putStrLn "\nCycle | Layer | Stage              | Ready | FFNDone  | WeightsRdy | WgtValid | LayerChange | WgtSample | Boot   | Token ID | Token"
  putStrLn "--------------------------------------------------------------------------------------------------------------------------------------"

  -- Loop through sampled outputs and display selected signals
  let printCycle (cycleIdx, tok) = do
        let li     = fromIntegral (layerIndicesSampled !! cycleIdx) :: Int
            ps     = processingStage (statesSampled !! cycleIdx)
            rdy    = readiesSampled !! cycleIdx
            ffn    = ffnDonesSampled !! cycleIdx
            wRdy   = weightsReadySampled !! cycleIdx
            wValid = weightValidSampled !! cycleIdx
            wSample = weightSampleSampled !! cycleIdx
            boot   = bootProgressSampled !! cycleIdx
            layChg = layerChangeSampled !! cycleIdx
            token  = coreOutputs !! cycleIdx
            sysSt  = sysStateSampled !! cycleIdx
            bootSt  = bootStateSampled !! cycleIdx
            emmcHS  = emmcHandshakesSampled !! cycleIdx
            ddrHS  = ddrHandshakesSampled !! cycleIdx
            emmcMasterRReady' = emmcMasterRReadySampled !! cycleIdx
        when (cycleIdx `mod` 1000 == 0 || rdy || ffn) $
          putStrLn $
            printf "%5d | %5d | %-18s | %5s | %8s | %8s | %10s | %8s | %9s |  %9d | %8s | %8s| %8s | %8s | %8s | %8s | %8s"
              cycleIdx
              li
              (show ps)
              (show boot)
              (show rdy)
              (show ffn)
              (show wRdy)
              (show wValid)
              (show layChg)
              wSample
              (show $ fst token)
              (show $ decodeToken tokenizer (fst token))
              (show sysSt)
              (show bootSt)
              (show emmcHS)
              (show ddrHS)
              (show emmcMasterRReady')

  mapM_ printCycle (zip [0 :: Int ..] coreOutputs)

  mapM_ (printToken tokenizer) sampledTokens
  putStrLn ""
  pure (length sampledTokens)

bundledOutputs :: DecoderParameters
  -> BSL.ByteString
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> ( C.Signal C.System (Token, Bool)
     , AxiMasterOut C.System
     , AxiMasterOut C.System
     , C.Signal C.System Bool
     , C.Signal C.System (C.Unsigned 32)
     , Top.DecoderIntrospection C.System
     ,  C.Signal C.System ReadState
     , C.Signal C.System ReadState
     )
bundledOutputs decoder modelBinary bundledInputs =
  (C.bundle (tokenOut, validOut), emmcMaster, ddrMaster
  , weightsReady, bootProgress
  , introspection
  , emmcReadState, ddrReadState
  )
 where
  (token, isValid, temperature, seed) = C.unbundle bundledInputs

  bypass = C.pure True  -- Use real AXI with file data
  powerOn = C.pure True

  -- Create the feedback loop: masters drive slaves, slaves feed back to decoder
  (tokenOut, validOut, emmcMaster, ddrMaster, weightsReady, bootProgress, introspection) =
    C.exposeClockResetEnable
      (Top.topEntityWithAxi bypass emmcSlave ddrSlave powerOn token isValid temperature seed)
      CS.systemClockGen
      CS.resetGen
      CS.enableGen

  -- Use file-backed AXI slaves that serve real model data
  -- For eMMC: serve the entire model file (used during boot to copy to DDR)
  (emmcSlave, emmcReadState) = C.exposeClockResetEnable
    (FileAxi.createFileBackedAxiSlave modelBinary emmcMaster)
    CS.systemClockGen
    CS.resetGen
    CS.enableGen

  -- For DDR: also serve from file (simulating that boot already copied data there)
  (ddrSlave, ddrReadState) = C.exposeClockResetEnable
    (FileAxi.createFileBackedAxiSlave modelBinary ddrMaster)
    CS.systemClockGen
    CS.resetGen
    CS.enableGen

bundledOutputs' :: DecoderParameters
  -> BSL.ByteString
  -> C.Signal C.System (Token, Bool, Temperature, Seed)
  -> ( C.Signal C.System (Token, Bool)
     , AxiMasterOut C.System
     , AxiMasterOut C.System
     , C.Signal C.System Bool
     , C.Signal C.System (C.Unsigned 32)
     , Top.DecoderIntrospection C.System
     , C.Signal C.System (C.Unsigned 32)
     , C.Signal C.System (C.Unsigned 32)
     , C.Signal C.System Bool
     , C.Signal C.System Bool
     )
bundledOutputs' decoder modelBinary bundledInputs =
  (C.bundle (tokenOut, validOut), emmcMaster, ddrMaster, weightsReady, bootProgress, introspection, pure 0, pure 0, pure False, pure False)
 where
  (token, isValid, temperature, seed) = C.unbundle bundledInputs
  
  bypass = C.pure True  -- CHANGED: Set to False to test real AXI
  powerOn = C.pure True
  
  -- Create the feedback loop: masters drive slaves, slaves feed back to decoder
  (tokenOut, validOut, emmcMaster, ddrMaster, weightsReady, bootProgress, introspection) =
    C.exposeClockResetEnable
      (Top.topEntityWithAxi bypass emmcSlave ddrSlave powerOn token isValid temperature seed)
      CS.systemClockGen
      CS.resetGen
      CS.enableGen
  
  -- Use file-backed AXI slaves that serve real model data
  -- For eMMC: serve the entire model file (used during boot to copy to DDR)
  emmcSlave = C.exposeClockResetEnable
    (SimAxi.createSimAxiSlave emmcMaster)
    CS.systemClockGen
    CS.resetGen
    CS.enableGen

  -- For DDR: also serve from file (simulating that boot already copied data there)
  ddrSlave = C.exposeClockResetEnable
    (SimAxi.createSimAxiSlave ddrMaster)
    CS.systemClockGen
    CS.resetGen
    CS.enableGen
