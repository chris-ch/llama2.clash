{-# LANGUAGE CPP #-}
module Simulation.ParamsPlaceholder (decoderConst, tokenizerFile, weightFile) where

import Prelude

import System.IO.Unsafe (unsafePerformIO)
import qualified Data.ByteString as BS (readFile)  -- Use STRICT ByteString
import qualified Data.Binary.Get as BG
import qualified Parser
import Simulation.Parameters (DecoderParameters)
import qualified Data.ByteString.Lazy.Char8 as BSL
import System.IO (hFlush, stdout)

-- Load weights with strict ByteString to avoid lazy I/O issues
{-# NOINLINE decoderConst #-}
decoderConst :: DecoderParameters
decoderConst = unsafePerformIO $ do
  putStrLn $ "Loading: " ++ weightFile
  content <- BS.readFile weightFile  -- STRICT, not lazy
  let params = BG.runGet Parser.parseLLaMa2ConfigFile (BSL.fromStrict content)
  putStrLn "✅ Simulation Model (FPGA wired) Loaded"
  hFlush stdout
  -- Force evaluation by pattern matching on the structure
  case params of
    p -> seq p (return p)

weightFile :: FilePath
weightFile =
#ifdef MODEL_NANO
  "./data/stories260K.bin"
#elif MODEL_260K
  "./data/stories260K.bin"
#elif MODEL_15M
  "./data/stories15M.bin"
#elif MODEL_42M
  "./data/stories42M.bin"
#elif MODEL_110M
  "./data/stories110M.bin"
#else
  "./data/stories260K.bin"
#endif

tokenizerFile :: FilePath
tokenizerFile =
#if defined(MODEL_NANO) || defined(MODEL_260K)
  "./data/tok512.bin"
#else
  "./data/tokenizer.bin"
#endif
