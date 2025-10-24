module LLaMa2.Layer.Attention.QKVProjectionIntegrationSpec (spec) where

import Clash.Prelude
import qualified Data.List as DL
import Test.Hspec
import qualified Prelude as P

import LLaMa2.Layer.Attention.QKVProjectionWeightBuffer
import LLaMa2.Memory.LayerAddressing (WeightAddress(..), WeightMatrixType(..))
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Types.ModelConfig
  ( ModelDimension
  , HeadDimension
  , NumQueryHeads
  , NumKeyValueHeads
  )
import Data.Maybe (fromJust)

-- ============================================================================
-- SIMULATION OF DECODER PATTERN
-- ============================================================================

decoderEnableCurrentPattern ::
  Signal System Bool  -- ^ streamValid
  -> Signal System WeightAddress
  -> Signal System Bool  -- ^ loadTrigger
  -> Signal System Bool  -- ^ enable (CURRENT - uses raw pulse)
decoderEnableCurrentPattern streamValid addr loadTrigger = enable
  where
    qkvDonePulse = isLastVWrite <$> addr .&&. streamValid
    enable = mux loadTrigger (pure False) qkvDonePulse

decoderEnableFixedPattern ::
  Signal System Bool  -- ^ streamValid
  -> Signal System WeightAddress
  -> Signal System QKVProjectionWeightBuffer
  -> Signal System Bool  -- ^ loadTrigger
  -> Signal System Bool  -- ^ enable (FIXED - uses fullyLoaded)
decoderEnableFixedPattern _streamValid _addr weightBuffer loadTrigger = enable
  where
    enable = mux loadTrigger (pure False) (fullyLoaded <$> weightBuffer)

-- Helper to check if address is last V write
isLastVWrite :: WeightAddress -> Bool
isLastVWrite WeightAddress{..} =
  matrixType == VMatrix
  && headIndex == fromInteger (natToNum @NumKeyValueHeads - 1)
  && rowIndex == maxBound

-- Create a test pattern with known signature
makeSyntheticRow :: Index HeadDimension -> RowI8E ModelDimension
makeSyntheticRow rowIdx =
  ( imap (\i _ -> fromIntegral (fromEnum rowIdx * 10 + fromEnum i)) (repeat (0 :: Signed 8))
  , fromIntegral (fromEnum rowIdx)
  )

-- Generate write sequence
-- Each weight matrix is HeadDimension x ModelDimension
-- We write HeadDimension rows for each head
generateWriteSequence ::
  Int ->
  [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
generateWriteSequence startCycle =
  qWrites P.++ kWrites P.++ vWrites
 where
  numQHeads  = fromInteger (natToNum @NumQueryHeads) :: Int
  numKVHeads = fromInteger (natToNum @NumKeyValueHeads) :: Int
  hdDim      = fromInteger (natToNum @HeadDimension) :: Int

  -- Q-matrix writes: numQHeads heads, each with hdDim rows
  qWrites =
    [ (writeCycle, True,
        WeightAddress rowIdx QMatrix (fromIntegral hd),
        makeSyntheticRow rowIdx)
    | hd <- [0 .. numQHeads - 1]  -- Head index (Integer)
    , rowIdx <- allIndices  -- Row index (Index HeadDimension)
    , let writeCycle = startCycle + hd * hdDim + fromEnum rowIdx
    ]
   where
    allIndices :: [Index HeadDimension]
    allIndices = [minBound .. maxBound]

  -- K-matrix writes
  kWrites =
    [ (writeCycle, True,
        WeightAddress rowIdx KMatrix (fromIntegral hd),
        makeSyntheticRow rowIdx)
    | let baseOffset = P.length qWrites
    , hd <- [0 .. numKVHeads - 1]
    , rowIdx <- allIndices
    , let writeCycle = startCycle + baseOffset + hd * hdDim + fromEnum rowIdx
    ]
   where
    allIndices :: [Index HeadDimension]
    allIndices = [minBound .. maxBound]

  -- V-matrix writes
  vWrites =
    [ (writeCycle, True,
        WeightAddress rowIdx VMatrix (fromIntegral hd),
        makeSyntheticRow rowIdx)
    | let baseOffset = P.length qWrites + P.length kWrites
    , hd <- [0 .. numKVHeads - 1]
    , rowIdx <- allIndices
    , let writeCycle = startCycle + baseOffset + hd * hdDim + fromEnum rowIdx
    ]
   where
    allIndices :: [Index HeadDimension]
    allIndices = [minBound .. maxBound]

createSignalsFromSequence ::
  Int
  -> [(Int, Bool, WeightAddress, RowI8E ModelDimension)]
  -> ( Signal System Bool
     , Signal System WeightAddress
     , Signal System (RowI8E ModelDimension)
     , Signal System Bool
     )
createSignalsFromSequence totalCycles writes =
  (streamValidSig, addrSig, rowSig, allDoneSig)
  where
    writeMap = DL.map (\(c, v, a, r) -> (c, (v, a, r))) writes
    defaultAddr = WeightAddress 0 QMatrix 0
    defaultRow = (repeat 0, 0)

    streamValidSig = fromList
      [ P.maybe False (\(v,_,_) -> v) (DL.lookup c writeMap)
      | c <- [0 .. totalCycles - 1]
      ]

    addrSig = fromList
      [ P.maybe defaultAddr (\(_,a,_) -> a) (DL.lookup c writeMap)
      | c <- [0 .. totalCycles - 1]
      ]

    rowSig = fromList
      [ P.maybe defaultRow (\(_,_,r) -> r) (DL.lookup c writeMap)
      | c <- [0 .. totalCycles - 1]
      ]

    allDoneSig = fromList
      [ case DL.lookup c writeMap of
          Just (True, addr, _) -> isLastVWrite addr
          _ -> False
      | c <- [0 .. totalCycles - 1]
      ]

-- ============================================================================
-- SPEC
-- ============================================================================

spec :: Spec
spec = do
  describe "Decoder enable signal timing (CURRENT pattern)" $ do
    it "demonstrates premature enable assertion" $ do
      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          totalCycles = lastWriteCycle + 500

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          loadTrigger = pure False
          reset = pure False

          -- Run buffer controller
          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          -- Generate enable using CURRENT pattern (raw pulse)
          enableCurrent = decoderEnableCurrentPattern streamValid addr loadTrigger

          -- Sample signals
          bufferSamples = DL.take totalCycles $ sample bufferSig
          enableSamples = DL.take totalCycles $ sample enableCurrent
          loadedFlags = DL.map fullyLoaded bufferSamples

          -- Find when enable first goes high
          firstEnableCycle = fromJust $ DL.findIndex id enableSamples

          -- Find when fullyLoaded first goes high
          firstLoadedCycle = fromJust $ DL.findIndex id loadedFlags

      -- Debug output
      P.putStrLn $ "Last write cycle: " P.++ show lastWriteCycle
      P.putStrLn $ "First enable cycle: " P.++ show firstEnableCycle
      P.putStrLn $ "First loaded cycle: " P.++ show firstLoadedCycle

      -- PROBLEM: Enable asserts at lastWriteCycle, but fullyLoaded is still False!
      firstEnableCycle `shouldBe` lastWriteCycle
      firstLoadedCycle `shouldBe` (lastWriteCycle + 1)

      -- At the cycle where enable first goes high, fullyLoaded is still False
      let bufferAtEnable = bufferSamples DL.!! firstEnableCycle
      fullyLoaded bufferAtEnable `shouldBe` False

  describe "Decoder enable signal timing (FIXED pattern)" $ do
    it "waits for fullyLoaded before enabling" $ do
      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          totalCycles = lastWriteCycle + 500

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          loadTrigger = pure False
          reset = pure False

          -- Run buffer controller
          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          -- Generate enable using FIXED pattern (fullyLoaded)
          enableFixed = decoderEnableFixedPattern streamValid addr bufferSig loadTrigger

          -- Sample signals
          bufferSamples = DL.take totalCycles $ sample bufferSig
          enableSamples = DL.take totalCycles $ sample enableFixed
          loadedFlags = DL.map fullyLoaded bufferSamples

          -- Find when enable first goes high
          firstEnableCycle = fromJust $ DL.findIndex id enableSamples

          -- Find when fullyLoaded first goes high
          firstLoadedCycle = fromJust $ DL.findIndex id loadedFlags

      -- FIXED: Enable asserts one cycle after last write
      firstEnableCycle `shouldBe` (lastWriteCycle + 1)
      firstLoadedCycle `shouldBe` (lastWriteCycle + 1)

      -- At the cycle where enable goes high, fullyLoaded is True
      let bufferAtEnable = bufferSamples DL.!! firstEnableCycle
      fullyLoaded bufferAtEnable `shouldBe` True

  describe "Write sequence generation" $ do
    it "generates correct number of writes" $ do
      let writes = generateWriteSequence 10
          numQHeads = fromInteger (natToNum @NumQueryHeads) :: Int
          numKVHeads = fromInteger (natToNum @NumKeyValueHeads) :: Int
          hdDim = fromInteger (natToNum @HeadDimension) :: Int
          
          expectedQWrites = numQHeads * hdDim
          expectedKWrites = numKVHeads * hdDim
          expectedVWrites = numKVHeads * hdDim
          expectedTotal = expectedQWrites + expectedKWrites + expectedVWrites

      P.length writes `shouldBe` expectedTotal

  describe "Data availability when enable asserts" $ do
    it "CURRENT pattern: data NOT guaranteed available when enable asserts" $ do
      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          totalCycles = lastWriteCycle + 500

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          loadTrigger = pure False
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          enableCurrent = decoderEnableCurrentPattern streamValid addr loadTrigger

          bufferSamples = DL.take totalCycles $ sample bufferSig
          enableSamples = DL.take totalCycles $ sample enableCurrent

          -- Find first cycle where enable is True
          firstEnableCycle = fromJust $ DL.findIndex id enableSamples

          -- Get buffer state at that cycle
          bufferAtEnable = bufferSamples DL.!! firstEnableCycle

      -- At this point, fullyLoaded should be False
      fullyLoaded bufferAtEnable `shouldBe` False

    it "FIXED pattern: data IS guaranteed available when enable asserts" $ do
      let writes = generateWriteSequence 10
          lastWriteCycle = P.maximum (DL.map (\(c,_,_,_) -> c) writes)
          totalCycles = lastWriteCycle + 500

          (streamValid, addr, row, allDone) = createSignalsFromSequence totalCycles writes
          loadTrigger = pure False
          reset = pure False

          bufferSig = withClockResetEnable systemClockGen resetGen enableGen $
            qkvWeightBufferController streamValid addr row allDone reset

          enableFixed = decoderEnableFixedPattern streamValid addr bufferSig loadTrigger

          bufferSamples = DL.take totalCycles $ sample bufferSig
          enableSamples = DL.take totalCycles $ sample enableFixed

          -- Find first cycle where enable is True
          firstEnableCycle = fromJust $ DL.findIndex id enableSamples

          -- Get buffer state at that cycle
          bufferAtEnable = bufferSamples DL.!! firstEnableCycle

          -- Read the LAST V matrix row (use maxBound which is in bounds)
          lastVRow = wvBuf (last (kvHeadBuffers bufferAtEnable)) !! (maxBound :: Index HeadDimension)
          (_, expValue) = lastVRow

      -- Now the data has been registered and is available
      fullyLoaded bufferAtEnable `shouldBe` True
      expValue `shouldNotBe` 0  -- Should have the synthetic signature
