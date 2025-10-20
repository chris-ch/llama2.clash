{-# LANGUAGE CPP #-}
module LLaMa2.ParamsPlaceholder (decoderConst) where

import Prelude

import System.IO.Unsafe (unsafePerformIO)
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Binary.Get as BG
import qualified Parser
import LLaMa2.Types.Parameters (DecoderParameters)


-- Load real trained weights at Clash compile-time
decoderConst :: DecoderParameters
decoderConst = unsafePerformIO $ do
  putStrLn "Loading weights for synthesis..."
  content <- BSL.readFile weightFile
  let params = BG.runGet Parser.parseLLaMa2ConfigFile content
  putStrLn $ "✅ Loaded " ++ weightFile
  return params
{-# NOINLINE decoderConst #-}

-- Select weight file based on CPP model flag
weightFile :: FilePath
weightFile = 
#ifdef MODEL_260K
  "./data/stories260K.bin"
#elif defined(MODEL_15M)
  "./data/stories15M.bin"
#elif defined(MODEL_42M)
  "./data/stories42M.bin"
#elif defined(MODEL_110M)
  "./data/stories110M.bin"
#else
  "./data/stories260K.bin"
#endif
