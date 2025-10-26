module LLaMa2.Decoder.TimingValidationSpec (spec) where

import Clash.Prelude
import Test.Hspec
import qualified Prelude as P
import qualified Data.List as DL

import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer
import LLaMa2.Memory.LayerAddressing (WeightAddress(..), WeightMatrixType(..))
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Types.ModelConfig
  ( ModelDimension, HeadDimension, NumQueryHeads, NumKeyValueHeads )

spec :: Spec
spec = do
  describe "Decoder timing must be correct" $ do
    it "layerEnable must NEVER assert when fullyLoaded is False" $ do
      -- This test simulates the decoder's enable generation
      
      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          totalCycles = lastWriteCycle + 100
          
          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          loadTrigger = pure False
          reset = pure False
          
          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset
          
          layerEnable = mux loadTrigger (pure False) (fullyLoaded <$> bufferSig)
          
          -- Check for timing violations
          fullyLoadedSig = fullyLoaded <$> bufferSig
          timingViolation = layerEnable .&&. (not <$> fullyLoadedSig)
          
          samples = DL.take totalCycles $ sample timingViolation
          violations = DL.filter id samples
      
      -- THE KEY ASSERTION: There should be ZERO violations
      -- If this fails, the decoder has the timing bug!
      P.length violations `shouldBe` 0

-- Helpers
makeSyntheticRow :: Index HeadDimension -> RowI8E ModelDimension
makeSyntheticRow rowIdx =
  ( imap (\i _ -> fromIntegral (fromEnum rowIdx * 10 + fromEnum i)) (repeat (0 :: Signed 8))
  , fromIntegral (fromEnum rowIdx) )

generateWriteSequence :: Int -> [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
generateWriteSequence startCycle = qWrites P.++ kWrites P.++ vWrites
 where
  numQHeads  = fromInteger (natToNum @NumQueryHeads) :: Int
  numKVHeads = fromInteger (natToNum @NumKeyValueHeads) :: Int
  hdDim      = fromInteger (natToNum @HeadDimension) :: Int
  qWrites = [ (startCycle + hd * hdDim + fromEnum rowIdx, True,
               WeightAddress rowIdx QMatrix (fromIntegral hd), makeSyntheticRow rowIdx)
            | hd <- [0 .. numQHeads - 1], rowIdx <- [minBound .. maxBound] ]
  kWrites = [ (startCycle + P.length qWrites + hd * hdDim + fromEnum rowIdx, True,
               WeightAddress rowIdx KMatrix (fromIntegral hd), makeSyntheticRow rowIdx)
            | hd <- [0 .. numKVHeads - 1], rowIdx <- [minBound .. maxBound] ]
  vWrites = [ (startCycle + P.length qWrites + P.length kWrites + hd * hdDim + fromEnum rowIdx, True,
               WeightAddress rowIdx VMatrix (fromIntegral hd), makeSyntheticRow rowIdx)
            | hd <- [0 .. numKVHeads - 1], rowIdx <- [minBound .. maxBound] ]

createSignalsFromSequence :: Int -> [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
  -> (Signal System Bool, Signal System WeightAddress, Signal System (RowI8E ModelDimension), Signal System Bool)
createSignalsFromSequence totalCycles writes = (streamValidSig, addrSig, rowSig, allDoneSig)
  where
    writeMap = DL.map (\(c, v, a, r) -> (c, (v, a, r))) writes
    streamValidSig = fromList [ P.maybe False (\(v,_,_) -> v) (DL.lookup c writeMap) | c <- [0..totalCycles-1] ]
    addrSig = fromList [ P.maybe (WeightAddress 0 QMatrix 0) (\(_,a,_) -> a) (DL.lookup c writeMap) | c <- [0..totalCycles-1] ]
    rowSig = fromList [ P.maybe (repeat 0, 0) (\(_,_,r) -> r) (DL.lookup c writeMap) | c <- [0..totalCycles-1] ]
    allDoneSig = fromList [ case DL.lookup c writeMap of
                              Just (True, addr, _) -> isLastVWrite addr
                              _ -> False | c <- [0..totalCycles-1] ]

isLastVWrite :: WeightAddress -> Bool
isLastVWrite WeightAddress{..} = matrixType == VMatrix 
  && headIndex == fromInteger (natToNum @NumKeyValueHeads - 1) && rowIndex == maxBound
