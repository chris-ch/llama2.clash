module Main (main) where

import Prelude

import qualified Data.Binary.Get as BG
import qualified Data.ByteString.Lazy as BSL
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy.Char8 as BSC
import qualified Data.List as DL
import qualified Options.Applicative as OA
import qualified Foreign as F
import qualified Data.Vector.Unboxed as V
import qualified Clash.Sized.Vector as CV
import qualified Clash.Prelude as C
import qualified Clash.Signal as CS

import GHC.IO (unsafePerformIO)
import Data.Maybe (fromMaybe)
import Data.Time.Clock.POSIX (getPOSIXTime)
import System.IO (hFlush, stdout)
import Control.Monad (replicateM_)
import Text.Printf (printf)
import Model.Core.Types
    (
      SingleHeadComponent(SingleHeadComponent, rotary, wqHead, wkHead,
                          wvHead),
      RotaryEncodingComponent(RotaryEncodingComponent, freqSin, freqCos),
      EmbeddingComponent(EmbeddingComponent, rmsFinalWeight, vocabulary),
      Token,
      CArray2D(CArray2D),
      Temperature, Seed )

import Model.Config
    (
      HeadDimension,
      SequenceLength,
      VocabularySize,
      NumQueryHeads,
      NumLayers,
      NumKeyValueHeads,
      HiddenDimension,
      ModelDimension,
      RotaryPositionalEmbeddingDimension
      )
import qualified Model.Top as Top ( topEntity )
import qualified Tokenizer as T (buildTokenizer, encodeTokens, Tokenizer, decodePiece)
import Model.Layers.TransformerLayer (TransformerDecoderComponent (..), TransformerLayerComponent (..))
import Model.Numeric.Types (FixedPoint)
import qualified Model.Layers.Components.Quantized as Quantized
    ( MultiHeadAttentionComponent, MultiHeadAttentionComponent(..), FeedForwardNetworkComponent(..), MultiHeadAttentionComponentQ, FeedForwardNetworkComponentQ, quantizeMHA, quantizeFFN, quantizeEmbedding )


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
    parseModel = BG.runGet parseModelConfigFile
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
    <*> OA.strOption (OA.long "tokenizer-file" <> OA.value "./data/tokenizer.bin" <> OA.help "Tokenizer binary file")
    <*> OA.strOption (OA.long "model-file" <> OA.value "./data/stories110M.bin" <> OA.metavar "MODEL_FILE" <> OA.help "Model binary file")
    <*> OA.option OA.auto (OA.long "temperature" <> OA.value 0.0 <> OA.metavar "TEMPERATURE" <> OA.help "Temperature")
    <*> OA.option OA.auto (OA.long "steps" <> OA.value 256 <> OA.metavar "STEPS" <> OA.help "Number of steps")
    <*> OA.optional (OA.strArgument (OA.metavar "PROMPT" <> OA.help "Initial prompt"))

-- ============================================================================
-- File Parsing
-- ============================================================================

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = do
  byteData <- BG.getByteString (count * 4)
  return $! unsafePerformIO $ do
    BS.useAsCString byteData $ \ptr -> do
      let floatPtr = F.castPtr ptr :: F.Ptr Float
      V.generateM count (F.peekElemOff floatPtr)

readVec1D :: forall n. C.KnownNat n => BG.Get (C.Vec n Float)
readVec1D = do
    let total = C.snatToNum (C.SNat :: C.SNat n)
    vec <- readVector total
    return $ CV.unsafeFromList (V.toList vec)

readVec2D :: forall n m. (C.KnownNat n, C.KnownNat m) => BG.Get (C.Vec n (C.Vec m Float))
readVec2D = do
    let n = C.snatToNum (C.SNat :: C.SNat n)
        m = C.snatToNum (C.SNat :: C.SNat m)
        total = n * m
    vec <- readVector total
    let floatList = V.toList vec
        chunks = chunksOf m floatList
        vecs = map CV.unsafeFromList chunks
    return $ CV.unsafeFromList vecs
  where
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

readVec3D :: forall n m p. (C.KnownNat n, C.KnownNat m, C.KnownNat p) => BG.Get (C.Vec n (C.Vec m (C.Vec p Float)))
readVec3D = do
    let n = C.snatToNum (C.SNat :: C.SNat n)
        m = C.snatToNum (C.SNat :: C.SNat m)
        p = C.snatToNum (C.SNat :: C.SNat p)
        total = n * m * p
    vec <- readVector total
    let floatList = V.toList vec
        innerChunks = chunksOf p floatList
        innerVecs = map CV.unsafeFromList innerChunks
        middleChunks = chunksOf m innerVecs
        middleVecs = map CV.unsafeFromList middleChunks
    return $ CV.unsafeFromList middleVecs
  where
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

readVec4D :: forall m n p q. (C.KnownNat m, C.KnownNat n, C.KnownNat p, C.KnownNat q) => BG.Get (C.Vec m (C.Vec n (C.Vec p (C.Vec q Float))))
readVec4D = do
    let m = C.snatToNum (C.SNat :: C.SNat m)
        n = C.snatToNum (C.SNat :: C.SNat n)
        p = C.snatToNum (C.SNat :: C.SNat p)
        q = C.snatToNum (C.SNat :: C.SNat q)
        total = m * n * p * q
    vec <- readVector total
    let floatList = V.toList vec
        -- First chunk into q-sized vectors
        innerChunks = chunksOf q floatList
        innerVecs = map CV.unsafeFromList innerChunks
        -- Then chunk into p groups of q-sized vectors
        middleChunks = chunksOf p innerVecs
        middleVecs = map CV.unsafeFromList middleChunks
        -- Then chunk into n groups of p×q-sized tensors
        outerChunks = chunksOf n middleVecs
        outerVecs = map CV.unsafeFromList outerChunks
        -- Finally chunk into m groups
    return $ CV.unsafeFromList outerVecs
  where
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

parseModelConfigFile :: BG.Get TransformerDecoderComponent
parseModelConfigFile = do
  replicateM_ 7 BG.getInt32le
  tokenEmbeddingTable' <- readVec2D @VocabularySize @ModelDimension
  rmsAttWeight'        <- readVec2D @NumLayers @ModelDimension
  wq'                  <- readVec4D @NumLayers @NumQueryHeads     @HeadDimension  @ModelDimension
  wk'                  <- readVec4D @NumLayers @NumKeyValueHeads  @HeadDimension  @ModelDimension
  wv'                  <- readVec4D @NumLayers @NumKeyValueHeads  @HeadDimension  @ModelDimension
  wo'                  <- readVec3D @NumLayers @ModelDimension    @ModelDimension
  rmsFfnWeight'        <- readVec2D @NumLayers @ModelDimension
  w1'                  <- readVec3D @NumLayers @HiddenDimension   @ModelDimension
  w2'                  <- readVec3D @NumLayers @ModelDimension    @HiddenDimension
  w3'                  <- readVec3D @NumLayers @HiddenDimension   @ModelDimension
  rmsFinalWeight'      <- readVec1D @ModelDimension
  freqCisReal'         <- readVec2D @SequenceLength @RotaryPositionalEmbeddingDimension
  freqCisImag'         <- readVec2D @SequenceLength @RotaryPositionalEmbeddingDimension

  let
    -- Build Float embedding, then quantize
    embeddingFloat = EmbeddingComponent
      { vocabulary     = CArray2D tokenEmbeddingTable'
      , rmsFinalWeight = rmsFinalWeight'
      }
    embeddingQ = Quantized.quantizeEmbedding embeddingFloat

    layer :: C.Index NumLayers -> TransformerLayerComponent
    layer lIdx =
      let
        sha :: C.Index NumQueryHeads -> SingleHeadComponent
        sha hIdx =
          let nQ  = C.snatToNum (C.SNat @NumQueryHeads)     :: Int
              nKV = C.snatToNum (C.SNat @NumKeyValueHeads)  :: Int
              kvMul = max 1 (nQ `div` nKV)
              kvIdxInt = fromIntegral hIdx `div` kvMul
              kvIdx :: C.Index NumKeyValueHeads
              kvIdx = fromInteger (toInteger kvIdxInt)
          in SingleHeadComponent
               { wqHead = CArray2D $ (wq' C.!! lIdx) C.!! hIdx
               , wkHead = CArray2D $ (wk' C.!! lIdx) C.!! kvIdx
               , wvHead = CArray2D $ (wv' C.!! lIdx) C.!! kvIdx
               , rotary  = RotaryEncodingComponent
                   { freqCos = CArray2D freqCisReal'
                   , freqSin = CArray2D freqCisImag'
                   }
               }
        woLayer :: C.Vec ModelDimension (C.Vec ModelDimension Float)
        woLayer = wo' C.!! lIdx
        headBlock :: C.Index NumQueryHeads -> CArray2D ModelDimension HeadDimension
        headBlock hIdx =
          let base :: Int
              base = fromIntegral hIdx * C.snatToNum (C.SNat @HeadDimension)
              rowSlice :: C.Vec ModelDimension Float -> C.Vec HeadDimension Float
              rowSlice row =
                C.map
                  (\off -> row C.!! (toEnum (base + fromIntegral off) :: C.Index ModelDimension))
                  (C.indicesI @HeadDimension)
          in CArray2D (C.map rowSlice woLayer)

        mWoVec :: C.Vec NumQueryHeads (CArray2D ModelDimension HeadDimension)
        mWoVec = C.map headBlock (C.indicesI @NumQueryHeads)

        mhaFloat :: Quantized.MultiHeadAttentionComponent
        mhaFloat = Quantized.MultiHeadAttentionComponent
          { heads  = C.map sha (C.indicesI :: C.Vec NumQueryHeads (C.Index NumQueryHeads))
          , mWo    = mWoVec
          , rmsAtt = rmsAttWeight' C.!! lIdx
          }

        mhaQ :: Quantized.MultiHeadAttentionComponentQ
        mhaQ = Quantized.quantizeMHA mhaFloat

        ffnFloat :: Quantized.FeedForwardNetworkComponent
        ffnFloat = Quantized.FeedForwardNetworkComponent
          { fW1     = CArray2D $ w1' C.!! lIdx
          , fW2     = CArray2D $ w2' C.!! lIdx
          , fW3     = CArray2D $ w3' C.!! lIdx
          , fRMSFfn = rmsFfnWeight' C.!! lIdx
          }
        ffnQ :: Quantized.FeedForwardNetworkComponentQ
        ffnQ = Quantized.quantizeFFN ffnFloat

      in TransformerLayerComponent
           { multiHeadAttention = mhaQ
           , feedforwardNetwork = ffnQ
           }

    decoder = TransformerDecoderComponent
      { modelEmbedding = embeddingQ
      , modelLayers    = C.map layer (C.indicesI :: C.Vec NumLayers (C.Index NumLayers))
      }

  return decoder

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
