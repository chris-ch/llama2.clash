{-# LANGUAGE CPP #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
module LLaMa2.ParamsPlaceholder (decoderConst) where

import Prelude

import System.IO.Unsafe (unsafePerformIO)
import qualified Data.ByteString as BS (readFile)  -- Use STRICT ByteString
import qualified Data.Binary.Get as BG
import qualified Parser
import LLaMa2.Types.Parameters (DecoderParameters)
import qualified Data.ByteString.Lazy.Char8 as BSL

-- Load weights with strict ByteString to avoid lazy I/O issues
{-# NOINLINE decoderConst #-}
decoderConst :: DecoderParameters
decoderConst = unsafePerformIO $ do
  putStrLn $ "Loading: " ++ weightFile
  content <- BS.readFile weightFile  -- STRICT, not lazy
  let params = BG.runGet Parser.parseLLaMa2ConfigFile (BSL.fromStrict content)
  putStrLn "✅ Loaded"
  -- Force evaluation by pattern matching on the structure
  case params of
    p -> seq p (return p)

weightFile :: FilePath
weightFile = 
#ifdef MODEL_260K
  "./data/stories260K.bin"
#else
  "./data/stories260K.bin"
#endif
