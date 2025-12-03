module LLaMa2.Layer.Attention.WeightLoaderDebugSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import LLaMa2.Layer.Attention.WeightLoader (weightLoader, WeightLoaderOutput(..))
import LLaMa2.Types.ModelConfig
import qualified Simulation.DRAMBackedAxiSlave as DRAMSlave
import qualified Simulation.Parameters as PARAM
import Test.Hspec
import qualified Prelude as P
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified Simulation.ParamsPlaceholder as PARAM
import Control.Monad (forM_, when, unless)
import Text.Printf (printf)
import Clash.Sized.Vector (unsafeFromList)
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Memory.AXI.Types (AxiAR(..))

type TestDRAMDepth = 65536

spec :: Spec
spec = do
  criticalRow1Test
  staticMemoryTests
  cycleByClycleTraceTests
  axiAddressDbg

-- ============================================================================
-- CRITICAL: Test the exact failing row FIRST
-- ============================================================================

criticalRow1Test :: Spec
criticalRow1Test = describe "WeightLoaderDbg - CRITICAL: Row 1 at address 131712" $ do

  it "static check: DRAM row 1 matches hardcoded row 1 (no simulation)" $ do
    let params = PARAM.decoderConst
        dramVec :: Vec TestDRAMDepth DRAMSlave.WordData
        dramVec = DRAMSlave.buildMemoryFromParams params

        -- Get hardcoded row 1 for Layer 0, Head 0
        layer0 = head (PARAM.modelLayers params)
        mha = PARAM.multiHeadAttention layer0
        qHead0 = PARAM.qMatrix (head (PARAM.qHeads mha))
        hcRow = qHead0 !! (1 :: Index HeadDimension)

        -- Calculate address and read from DRAM
        addr = Layout.rowAddressCalculator Layout.QMatrix 0 0 1
        wordIdx = fromIntegral (addr `shiftR` 6) :: Int
        word0 = dramVec !! wordIdx
        word1 = dramVec !! (wordIdx + 1)
        wordsVec = word0 :> word1 :> Nil :: Vec 2 (BitVector 512)
        dramRow = Layout.multiWordRowParser @ModelDimension wordsVec

    P.putStrLn "\n=== CRITICAL ROW 1 CHECK ==="
    P.putStrLn $ "Address: " P.++ show addr P.++ " (expected 131712)"
    P.putStrLn $ "Word indices: " P.++ show wordIdx P.++ ", " P.++ show (wordIdx + 1)
    P.putStrLn ""
    P.putStrLn "HC row 1:"
    P.putStrLn $ "  exp=" P.++ show (rowExponent hcRow)
    P.putStrLn $ "  mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas hcRow)
    P.putStrLn ""
    P.putStrLn "DRAM row 1:"
    P.putStrLn $ "  exp=" P.++ show (rowExponent dramRow)
    P.putStrLn $ "  mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas dramRow)
    P.putStrLn ""
    P.putStrLn $ "Match: " P.++ show (hcRow == dramRow)

    addr `shouldBe` 131712
    hcRow `shouldBe` dramRow

  it "static check: row 0 also matches (sanity check)" $ do
    let params = PARAM.decoderConst
        dramVec :: Vec TestDRAMDepth DRAMSlave.WordData
        dramVec = DRAMSlave.buildMemoryFromParams params

        layer0 = head (PARAM.modelLayers params)
        mha = PARAM.multiHeadAttention layer0
        qHead0 = PARAM.qMatrix (head (PARAM.qHeads mha))
        hcRow0 = qHead0 !! (0 :: Index HeadDimension)

        addr0 = Layout.rowAddressCalculator Layout.QMatrix 0 0 0
        wordIdx0 = fromIntegral (addr0 `shiftR` 6) :: Int
        words0 = (dramVec !! wordIdx0) :> (dramVec !! (wordIdx0 + 1)) :> Nil
        dramRow0 = Layout.multiWordRowParser @ModelDimension words0

    P.putStrLn "\n=== ROW 0 SANITY CHECK ==="
    P.putStrLn $ "Address: " P.++ show addr0
    P.putStrLn $ "HC mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas hcRow0)
    P.putStrLn $ "DRAM mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas dramRow0)
    P.putStrLn $ "Match: " P.++ show (hcRow0 == dramRow0)

    hcRow0 `shouldBe` dramRow0

  it "compares row 0 and row 1 data (are they different?)" $ do
    let params = PARAM.decoderConst
        layer0 = head (PARAM.modelLayers params)
        mha = PARAM.multiHeadAttention layer0
        qHead0 = PARAM.qMatrix (head (PARAM.qHeads mha))
        hcRow0 = qHead0 !! (0 :: Index HeadDimension)
        hcRow1 = qHead0 !! (1 :: Index HeadDimension)

    P.putStrLn "\n=== ROW 0 vs ROW 1 (should be different) ==="
    P.putStrLn $ "Row 0 mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas hcRow0)
    P.putStrLn $ "Row 1 mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas hcRow1)
    P.putStrLn $ "Same: " P.++ show (hcRow0 == hcRow1)

    -- Rows should be different (unless model has degenerate weights)
    hcRow0 `shouldNotBe` hcRow1

  it "checks if error values match a different row" $ do
    -- The error showed DRAM mant = [5,69,58,43,-11,6,-11,-24]
    -- Let's find if this matches any row in the Q matrix
    let params = PARAM.decoderConst
        layer0 = head (PARAM.modelLayers params)
        mha = PARAM.multiHeadAttention layer0

        errorMants = [5, 69, 58, 43, -11, 6, -11, -24] :: [Signed 8]

        findMatchingRow =
          [ (hi, ri, P.take 8 $ toList $ rowMantissas row)
          | hi <- [0..natToNum @NumQueryHeads - 1 :: Int]
          , let qHead = PARAM.qMatrix (PARAM.qHeads mha !! hi)
          , ri <- [0..natToNum @HeadDimension - 1 :: Int]
          , let row = qHead !! ri
          , P.take 8 (toList $ rowMantissas row) == errorMants
          ]

    P.putStrLn "\n=== SEARCHING FOR ERROR DATA ==="
    P.putStrLn $ "Looking for mant[0..7]=" P.++ show errorMants

    case findMatchingRow of
      [] -> P.putStrLn "No matching row found in Layer 0 Q matrices"
      matches -> do
        P.putStrLn $ "FOUND " P.++ show (P.length matches) P.++ " matching row(s):"
        forM_ matches $ \(hi, ri, mants) ->
          P.putStrLn $ "  Head " P.++ show hi P.++ ", Row " P.++ show ri P.++
                       " mant[0..7]=" P.++ show mants

    -- This test is informational - always passes
    True `shouldBe` True

-- ============================================================================
-- Static Memory Tests (no simulation, just data comparison)
-- ============================================================================

staticMemoryTests :: Spec
staticMemoryTests = describe "WeightLoaderDbg - Static Memory Verification (pre-simulation)" $ do

  it "verifies params identity (same object)" $ do
    let params1 = PARAM.decoderConst
        params2 = PARAM.decoderConst
        -- Extract a sample value from each to verify they're equivalent
        layer0_1 = head (PARAM.modelLayers params1)
        layer0_2 = head (PARAM.modelLayers params2)
        mha1 = PARAM.multiHeadAttention layer0_1
        mha2 = PARAM.multiHeadAttention layer0_2
        qHead0_1 = PARAM.qMatrix (head (PARAM.qHeads mha1))
        qHead0_2 = PARAM.qMatrix (head (PARAM.qHeads mha2))
        row0_1 = qHead0_1 !! (0 :: Index HeadDimension)
        row0_2 = qHead0_2 !! (0 :: Index HeadDimension)
    row0_1 `shouldBe` row0_2

  it "verifies ALL Q matrix rows in DRAM match hardcoded (Layer 0, all heads)" $ do
    let params = PARAM.decoderConst
        dramVec :: Vec TestDRAMDepth DRAMSlave.WordData
        dramVec = DRAMSlave.buildMemoryFromParams params

        -- Check every row of every head in layer 0
        checkAllHeads =
          [ checkRow params dramVec li hi ri
          | li <- [0 :: Int]  -- Layer 0 only for now
          , hi <- [0..natToNum @NumQueryHeads - 1]
          , ri <- [0..natToNum @HeadDimension - 1]
          ]

    P.putStrLn "\n=== Checking ALL Q matrix rows (Layer 0) ==="
    results <- P.mapM (\(li, hi, ri, match, hcRow, dramRow) -> do
      unless match $ do
        P.putStrLn $ printf "MISMATCH: Layer %d, Head %d, Row %d" li hi ri
        P.putStrLn $ "  HC exp=" P.++ show (rowExponent hcRow) P.++
                     " mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas hcRow)
        P.putStrLn $ "  DRAM exp=" P.++ show (rowExponent dramRow) P.++
                     " mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas dramRow)
      return match
      ) checkAllHeads

    P.and results `shouldBe` True

  it "verifies ALL Q matrix rows across ALL layers" $ do
    let params = PARAM.decoderConst
        dramVec :: Vec TestDRAMDepth DRAMSlave.WordData
        dramVec = DRAMSlave.buildMemoryFromParams params

        checkAllLayers =
          [ checkRow params dramVec li hi ri
          | li <- [0..natToNum @NumLayers - 1]
          , hi <- [0..natToNum @NumQueryHeads - 1]
          , ri <- [0..natToNum @HeadDimension - 1]
          ]

    P.putStrLn $ "\n=== Checking ALL Q matrix rows (all " P.++
                 show (natToNum @NumLayers :: Int) P.++ " layers) ==="
    P.putStrLn $ "Total rows to check: " P.++ show (P.length checkAllLayers)

    results <- P.mapM (\(li, hi, ri, match, hcRow, dramRow) -> do
      unless match $ do
        P.putStrLn $ printf "MISMATCH: Layer %d, Head %d, Row %d" li hi ri
        let addr = Layout.rowAddressCalculator Layout.QMatrix
                     (toEnum li) (toEnum hi) (toEnum ri)
        P.putStrLn $ "  Address: " P.++ show addr
        P.putStrLn $ "  HC exp=" P.++ show (rowExponent hcRow) P.++
                     " mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas hcRow)
        P.putStrLn $ "  DRAM exp=" P.++ show (rowExponent dramRow) P.++
                     " mant[0..7]=" P.++ show (P.take 8 $ toList $ rowMantissas dramRow)
      return match
      ) checkAllLayers

    let mismatches = P.length $ P.filter not results
    P.putStrLn $ "Mismatches found: " P.++ show mismatches P.++ " / " P.++ show (P.length results)
    P.and results `shouldBe` True

  it "dumps address map for Layer 0, Head 0" $ do
    P.putStrLn "\n=== Address Map (Layer 0, Head 0) ==="
    P.putStrLn "Row | Address  | Word"
    forM_ [0..natToNum @HeadDimension - 1 :: Int] $ \ri -> do
      let addr = Layout.rowAddressCalculator Layout.QMatrix 0 0 (toEnum ri)
          word = addr `div` 64
      P.putStrLn $ printf "%3d | %8d | %4d" ri addr word
    True `shouldBe` True

-- Helper function for static row checking
checkRow :: PARAM.DecoderParameters
         -> Vec TestDRAMDepth DRAMSlave.WordData
         -> Int -> Int -> Int
         -> (Int, Int, Int, Bool, RowI8E ModelDimension, RowI8E ModelDimension)
checkRow params dramVec li hi ri =
  let layer = PARAM.modelLayers params !! li
      mha = PARAM.multiHeadAttention layer
      qHead = PARAM.qHeads mha !! hi
      qMat = PARAM.qMatrix qHead
      hcRow = qMat !! ri

      addr = Layout.rowAddressCalculator Layout.QMatrix
               (toEnum li) (toEnum hi) (toEnum ri)
      baseWord = fromIntegral (addr `shiftR` 6) :: Int

      wordsPerRow = Layout.wordsPerRowVal @ModelDimension
      wordsList = [dramVec !! (baseWord + k) | k <- [0..wordsPerRow-1]]
      wordsVec = unsafeFromList wordsList :: Vec (Layout.WordsPerRow ModelDimension) (BitVector 512)

      dramRow = Layout.multiWordRowParser wordsVec
      match = hcRow == dramRow
  in (li, hi, ri, match, hcRow, dramRow)

-- ============================================================================
-- Cycle-by-Cycle Trace Tests
-- ============================================================================

cycleByClycleTraceTests :: Spec
cycleByClycleTraceTests = describe "WeightLoaderDbg - Cycle-by-Cycle Trace" $ do

  it "traces first 4 row fetches in detail" $ do
    let params = PARAM.decoderConst
        maxCycles = 250
        cyclesPerRequest = 50

        -- Request rows 0, 1, 2, 3
        requestGroups = [(0, False)] :
              [(toEnum i, True) : P.replicate (cyclesPerRequest - 1) (toEnum i, False)
              | i <- [0..3::Int]]
        requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

        reqSig = fromList (P.map fst requestPairs P.++ P.repeat 0)
        reqValidSig = fromList (P.map snd requestPairs P.++ P.repeat False)
        readySig = pure True

        dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
        dramContents = DRAMSlave.buildMemoryFromParams params

        realDRAM masterOut' =
          exposeClockResetEnable
            (DRAMSlave.createDRAMBackedAxiSlaveFromVec
               (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
            CS.systemClockGen CS.resetGen CS.enableGen

        (axiDRAM, outDRAM, dvDRAM, readyOut) =
          exposeClockResetEnable
            (weightLoader (realDRAM axiDRAM) 0 0 reqSig reqValidSig readySig (pure True) params)
            CS.systemClockGen CS.resetGen CS.enableGen

        -- Sample all debug signals
        validsSampled = sampleN maxCycles dvDRAM
        readysSampled = sampleN maxCycles readyOut
        statesSampled = sampleN maxCycles (dbgLoadState outDRAM)
        triggersSampled = sampleN maxCycles (dbgFetchTrigger outDRAM)
        fetchValidsSampled = sampleN maxCycles (dbgMultiWordValid outDRAM)
        capRowsSampled = sampleN maxCycles (dbgCapturedRowReq outDRAM)
        capAddrsSampled = sampleN maxCycles (dbgCapturedAddr outDRAM)
        liveAddrsSampled = sampleN maxCycles (dbgRequestedAddr outDRAM)
        reqsSampled = sampleN maxCycles reqSig
        reqValidsSampled = sampleN maxCycles reqValidSig

    P.putStrLn "\n=== Cycle-by-Cycle Trace (rows 0-3) ==="
    P.putStrLn "Cyc | Req | RqV | Rdy | State    | Trig | FVal | DV  | CapRow | CapAddr | LiveAddr"
    P.putStrLn (P.replicate 95 '-')

    forM_ [0..99] $ \n -> do
      let req = reqsSampled P.!! n
          rqv = reqValidsSampled P.!! n
          rdy = readysSampled P.!! n
          st = statesSampled P.!! n
          trig = triggersSampled P.!! n
          fval = fetchValidsSampled P.!! n
          dv = validsSampled P.!! n
          capRow = capRowsSampled P.!! n
          capAddr = capAddrsSampled P.!! n
          liveAddr = liveAddrsSampled P.!! n

      -- Only print interesting cycles (state changes or signals active)
      when (trig || fval || dv || rqv || n < 5) $
        P.putStrLn $ printf "%3d | %3d | %3s | %3s | %8s | %4s | %4s | %3s | %6d | %7d | %8d"
          n (fromEnum req :: Int) (show rqv) (show rdy) (show st)
          (show trig) (show fval) (show dv)
          (fromEnum capRow :: Int) capAddr liveAddr

    -- Verify all 4 completed
    let validCycles = [n | n <- [0..maxCycles-1], validsSampled P.!! n]
    P.putStrLn $ "\nValid output cycles: " P.++ show validCycles
    P.length validCycles `shouldSatisfy` (>= 4)

axiAddressDbg :: Spec
axiAddressDbg = describe "WeightLoaderDbg - AXI Address Debug" $ do

    it "traces actual AXI AR addresses vs expected" $ do
      let params = PARAM.decoderConst
          maxCycles = 300
          cyclesPerRequest = 30

          -- Request rows 0, 1, 2 with spacing
          requestGroups = [(0, False)] :
                [(toEnum i, True) : P.replicate (cyclesPerRequest - 1) (toEnum i, False)
                | i <- [0..2::Int]]
          requestPairs = P.concat requestGroups P.++ P.repeat (0, False)

          reqSig = fromList (P.map fst requestPairs P.++ P.repeat 0)
          reqValidSig = fromList (P.map snd requestPairs P.++ P.repeat False)
          readySig = pure True

          dramContents :: Vec TestDRAMDepth DRAMSlave.WordData
          dramContents = DRAMSlave.buildMemoryFromParams params

          realDRAM masterOut' =
            exposeClockResetEnable
              (DRAMSlave.createDRAMBackedAxiSlaveFromVec
                (DRAMSlave.DRAMConfig 1 0 1) dramContents masterOut')
              CS.systemClockGen CS.resetGen CS.enableGen

          (axiMaster, outDRAM, dvDRAM, readyOut) =
            exposeClockResetEnable
              (weightLoader (realDRAM axiMaster) 0 0 reqSig reqValidSig readySig (pure True) params)
              CS.systemClockGen CS.resetGen CS.enableGen

          -- Sample AXI AR channel
          arValidSampled = sampleN maxCycles (Master.arvalid axiMaster)
          arAddrSampled = sampleN maxCycles (araddr <$> Master.ardata axiMaster)

          -- Sample debug signals
          liveAddrSampled = sampleN maxCycles (dbgRequestedAddr outDRAM)
          capAddrSampled = sampleN maxCycles (dbgCapturedAddr outDRAM)
          triggerSampled = sampleN maxCycles (dbgFetchTrigger outDRAM)
          fetchValidSampled = sampleN maxCycles (dbgMultiWordValid outDRAM)
          dvSampled = sampleN maxCycles dvDRAM
          capRowSampled = sampleN maxCycles (dbgCapturedRowReq outDRAM)

      P.putStrLn "\n=== AXI Address Trace ==="
      P.putStrLn "Expected addresses:"
      P.putStrLn $ "  Row 0: " P.++ show (Layout.rowAddressCalculator Layout.QMatrix 0 0 0)
      P.putStrLn $ "  Row 1: " P.++ show (Layout.rowAddressCalculator Layout.QMatrix 0 0 1)
      P.putStrLn $ "  Row 2: " P.++ show (Layout.rowAddressCalculator Layout.QMatrix 0 0 2)
      P.putStrLn $ "  Head 1 Row 0: " P.++ show (Layout.rowAddressCalculator Layout.QMatrix 0 1 0)
      P.putStrLn ""

      P.putStrLn "Cyc | ARval | AR addr | LiveAddr | CapAddr | Trig | FVal | DV | CapRow"
      P.putStrLn (P.replicate 80 '-')

      forM_ [0..99] $ \n -> do
        let arv = arValidSampled P.!! n
            ara = arAddrSampled P.!! n
            live = liveAddrSampled P.!! n
            cap = capAddrSampled P.!! n
            trig = triggerSampled P.!! n
            fval = fetchValidSampled P.!! n
            dv = dvSampled P.!! n
            row = capRowSampled P.!! n

        when (arv || trig || fval || dv || n < 5) $
          P.putStrLn $ printf "%3d | %5s | %7d | %8d | %7d | %4s | %4s | %2s | %d"
            n (show arv) ara live cap (show trig) (show fval) (show dv) (fromEnum row :: Int)

      -- Find cycles where AR was valid
      let arValidCycles = [(n, arAddrSampled P.!! n) | n <- [0..maxCycles-1], arValidSampled P.!! n]
      P.putStrLn ""
      P.putStrLn "=== AR Valid Events ==="
      forM_ arValidCycles $ \(n, addr) -> do
        -- Determine which row this address corresponds to
        let row0Addr = Layout.rowAddressCalculator Layout.QMatrix 0 0 0
            row1Addr = Layout.rowAddressCalculator Layout.QMatrix 0 0 1
            row2Addr = Layout.rowAddressCalculator Layout.QMatrix 0 0 2
            h1r0Addr = Layout.rowAddressCalculator Layout.QMatrix 0 1 0
            label | addr == row0Addr = ("Head 0 Row 0 ✓" :: String)
                  | addr == row1Addr = "Head 0 Row 1 ✓"
                  | addr == row2Addr = "Head 0 Row 2 ✓"
                  | addr == h1r0Addr = "*** HEAD 1 ROW 0 - WRONG! ***"
                  | otherwise = "UNKNOWN"
        P.putStrLn $ printf "  Cycle %d: AR addr=%d -> %s" n addr label

      True `shouldBe` True
