module LLaMa2.Numeric.RmsNormSeqSpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Numeric.RmsNormSeq (rmsNormSeq)
import LLaMa2.Numeric.Types (FixedPoint)
import Test.Hspec
import qualified Prelude as P

-- ---------------------------------------------------------------------------
-- n = 8 throughout (MODEL_NANO dimension, fast to simulate)
-- Latency = 2*8 + 2 = 18 cycles from validIn to first outputValid.
-- ---------------------------------------------------------------------------

type N = 8

-- | Run rmsNormSeq for 'cycles' steps and return (validOut stream, result stream).
runSim
  :: [Bool]                  -- validIn stream
  -> [Vec N FixedPoint]      -- x stream
  -> [Vec N FixedPoint]      -- w stream
  -> Int                     -- cycles to sample
  -> [(Bool, Vec N FixedPoint)]
runSim validIns xs ws cycles =
  P.take cycles $ P.zip valids results
 where
  (validSig, resultSig, _) =
    exposeClockResetEnable
      (rmsNormSeq
        (fromList validIns)
        (fromList xs)
        (fromList ws))
      CS.systemClockGen
      CS.resetGen
      CS.enableGen
  valids  = sample validSig
  results = sample resultSig

-- | Reference result using the combinational version.
reference :: Vec N FixedPoint -> Vec N FixedPoint -> Vec N FixedPoint
reference = rmsNormFwFix

tolerance :: FixedPoint
tolerance = 0.01

withinTol :: Vec N FixedPoint -> Vec N FixedPoint -> Bool
withinTol a b = P.all (\(x, y) -> abs (x - y) < tolerance) (P.zip (toList a) (toList b))

spec :: Spec
spec = describe "RmsNormSeq" $ do

  describe "single input" $ do
    let x = 0.1 :> 0.2 :> 0.3 :> 0.4 :> 0.5 :> 0.6 :> 0.7 :> 0.8 :> Nil :: Vec N FixedPoint
        w = 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> Nil :: Vec N FixedPoint
        -- validIn fires at cycle 0 only; x/w held constant
        validIns = True : P.repeat False
        xs       = P.repeat x
        ws       = P.repeat w
        latency  = 2 * (8 :: Int) + 2  -- 18
        sim      = runSim validIns xs ws (latency + 5)
        expected = reference x w

    it "outputValid is False before latency" $
      P.all (not . P.fst) (P.take (latency - 1) sim) `shouldBe` True

    it "outputValid goes True at cycle 18" $
      P.fst (sim P.!! latency) `shouldBe` True

    it "result matches rmsNormFwFix within tolerance" $
      withinTol (P.snd (sim P.!! latency)) expected `shouldBe` True

    it "outputValid stays True after latency" $
      P.all P.fst (P.drop latency sim) `shouldBe` True

  -- Regression test for the premature-clear bug: rmsNormValid is still True
  -- from run 1 when run 2's validIn fires.  If pendingInput clears on
  -- rmsNormValid (not on effectiveValidIn) the second run is skipped.
  describe "second input fired while outputValid still True from first" $ do
    let x1 = 0.1 :> 0.2 :> 0.3 :> 0.4 :> 0.5 :> 0.6 :> 0.7 :> 0.8 :> Nil :: Vec N FixedPoint
        x2 = 0.9 :> 0.8 :> 0.7 :> 0.6 :> 0.5 :> 0.4 :> 0.3 :> 0.2 :> Nil :: Vec N FixedPoint
        w  = 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> 1.0 :> Nil :: Vec N FixedPoint
        latency = 2 * (8 :: Int) + 2  -- 18
        -- trigger2 = latency + 1: second validIn fires one cycle AFTER first Done,
        -- so outputValid from run 1 is still True when run 2's validIn arrives.
        trigger2 = latency + 1
        validIns = [ i == 0 || i == trigger2 | i <- [0..] ]
        xs       = [ if i < trigger2 then x1 else x2 | i <- [0..] ]
        ws       = P.repeat w
        sim      = runSim validIns xs ws (trigger2 + latency + 5)
        expected2 = reference x2 w

    it "second outputValid fires at trigger2 + latency" $
      P.fst (sim P.!! (trigger2 + latency)) `shouldBe` True

    it "second result matches rmsNormFwFix within tolerance" $
      withinTol (P.snd (sim P.!! (trigger2 + latency))) expected2 `shouldBe` True
