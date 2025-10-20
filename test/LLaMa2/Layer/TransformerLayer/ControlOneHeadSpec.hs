module LLaMa2.Layer.TransformerLayer.ControlOneHeadSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import Test.Hspec
import qualified Prelude as P

import LLaMa2.Config (ModelDimension, HeadDimension)
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent)
import LLaMa2.Numeric.ParamPack (MatI8E, RowI8E, dequantRowToF)
import LLaMa2.Layer.TransformerLayer.Internal (singleHeadController)
import LLaMa2.Helpers.FixedPoint (dotProductF)

-- ==========
-- Fixtures
-- ==========

-- Deterministic WO matrix:
-- each output row uses mantissas [1..HeadDimension], exponent 0
makeWO :: MatI8E ModelDimension HeadDimension
makeWO = 
  let rowMant :: Vec HeadDimension Mantissa
      rowMant = map (fromIntegral . (1 +) . fromEnum) (indicesI @HeadDimension)
      oneRow :: (Vec HeadDimension Mantissa, Exponent)
      oneRow = (rowMant, 0)
  in repeat oneRow

-- Deterministic head vector: x_j = 1/(j+1)
makeHeadVec :: Vec HeadDimension FixedPoint
makeHeadVec =
  map (\j -> realToFrac (1.0 / (1.0 + fromIntegral (fromEnum j) :: Double)))
      (indicesI @HeadDimension)

-- Dot product: dequantize a row once, then reuse existing F dot-product.
dotProductRowI8E :: KnownNat n => RowI8E n -> Vec n FixedPoint -> FixedPoint
dotProductRowI8E row = dotProductF (dequantRowToF row)

-- Matrix @ vector where matrix is quantized (I8E rows) and vector is FixedPoint.
matrixVectorMult
  :: (KnownNat cols)
  => MatI8E rows cols
  -> Vec cols FixedPoint
  -> Vec rows FixedPoint
matrixVectorMult byRows xF =
  map (`dotProductRowI8E` xF) byRows

-- Golden projected vector using combinational kernel
goldenWOx :: Vec ModelDimension FixedPoint
goldenWOx = matrixVectorMult makeWO makeHeadVec

withinTolVec :: FixedPoint -> Vec ModelDimension FixedPoint -> Vec ModelDimension FixedPoint -> Bool
withinTolVec tol a b =
  let diffs = P.zipWith (\x y -> abs (x - y)) (toList a) (toList b)
  in P.all (< tol) diffs

-- Conservative latency bound for sequentialMatVec
worstLatency :: Int
worstLatency =
  let rows = natToNum @ModelDimension :: Int
      cols = natToNum @HeadDimension  :: Int
  in rows * (cols + 2) + 16

-- ==========
-- DUT wrapper
-- ==========

-- Expose singleHeadController
singleHeadControllerDUT
  :: HiddenClockResetEnable System
  => Signal System Bool                         -- validIn (start request)
  -> ( Signal System (Vec ModelDimension FixedPoint) -- projOut (held at valid)
     , Signal System Bool                       -- validOut (pulse)
     , Signal System Bool )                     -- readyOut (level)
singleHeadControllerDUT validIn =
  singleHeadController validIn (pure makeHeadVec) makeWO

-- ==========
-- Simulation helpers
-- ==========

-- Drive a single start pulse at a chosen cycle
runSingleStart ::
  Int ->                     -- total cycles to sample
  Int ->                     -- cycle to fire (>=1 recommended)
  ( [Vec ModelDimension FixedPoint]  -- projOut samples
  , [Bool]                           -- validOut samples
  , [Bool] )                         -- readyOut samples
runSingleStart n fireAt =
  let pulses = P.replicate fireAt False P.++ [True] P.++ DL.repeat False
      headDoneSig = fromList pulses
      (projSig, validSig, readySig) =
        exposeClockResetEnable singleHeadControllerDUT
          CS.systemClockGen CS.resetGen CS.enableGen
          headDoneSig
  in ( DL.take n (sample projSig)
     , DL.take n (sample validSig)
     , DL.take n (sample readySig)
     )

-- Closed-loop handshake driver:
-- asserts headDone when readyOut=1 and we still need more completions.
-- Counts completions by watching validOut pulses.
runNHandshaked ::
  Int ->  -- N completions to request
  Int ->  -- max cycles to simulate (must be >= N*worstLatency + margin)
  ( [Vec ModelDimension FixedPoint]  -- projOut samples
  , [Bool]                           -- validOut samples
  , [Bool] )                         -- readyOut samples
runNHandshaked nReq maxCycles =
  let -- feedback driver: state = number of completions seen
      driverT :: Unsigned 32 -> (Bool, Bool) -> (Unsigned 32, Bool)
      driverT doneCnt (ready, valid) =
        let doneCnt' = if valid then doneCnt + 1 else doneCnt
            fire     = ready && doneCnt < fromIntegral nReq
        in (doneCnt', fire)

      -- We build a closed loop where the driver looks at (ready,valid)
      -- and emits 'headDone' pulses. Mealy provides the needed register.
      driver :: HiddenClockResetEnable System => Signal System Bool
      driver = mealy driverT 0 (bundle (readySig, validSig))

      -- Instantiate DUT and tie the loop
      (projSig, validSig, readySig) =
        exposeClockResetEnable
          (\() ->
             let headDoneSig = driver
             in singleHeadControllerDUT headDoneSig)
          CS.systemClockGen CS.resetGen CS.enableGen
          ()

  in ( DL.take maxCycles (sample projSig)
     , DL.take maxCycles (sample validSig)
     , DL.take maxCycles (sample readySig)
     )

collectValidEvents :: [Vec ModelDimension FixedPoint]
  -> [Bool]
  -> [(Int, Vec ModelDimension FixedPoint)]
collectValidEvents outs valids =
  [ (i, o) | (i,(v,o)) <- P.zip [0..] (P.zip valids outs), v ]

-- ==========
-- Spec
-- ==========

spec :: Spec
spec = do
  describe "singleHeadController - handshaked behavior" $ do
    context "produces correct projection for a single start pulse" $ do
      it "produces correct projection for a single start pulse" $ do
        let
          maxCycles = worstLatency + 32
          fireAt    = 1
          (outs, valids, _readys) = runSingleStart maxCycles fireAt
          events    = collectValidEvents outs valids
          (_, got) = P.head events
          tol      = 0.01
          
        P.length (DL.elemIndices True valids) `shouldSatisfy` (== 1)
        withinTolVec tol got goldenWOx `shouldBe` True

      it "accepts exactly N handshaked starts and produces N completions with correct results" $ do
        let nReq      = 3
            maxCycles = nReq * worstLatency + 64
            (outs, valids, _readys) = runNHandshaked nReq maxCycles
            events    = collectValidEvents outs valids
        P.length events `shouldBe` nReq
        let tol = 0.01
        P.all (\(_, got) -> withinTolVec tol got goldenWOx) events `shouldBe` True

      it "readyOut toggles (not stuck) during an operation" $ do
        let maxCycles = worstLatency + 32
            fireAt    = 1
            (_outs, _valids, readys) = runSingleStart maxCycles fireAt
        (P.length (DL.nub (DL.take maxCycles readys)) > 1) `shouldBe` True
