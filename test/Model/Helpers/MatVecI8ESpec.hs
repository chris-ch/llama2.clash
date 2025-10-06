module Model.Helpers.MatVecI8ESpec (spec) where

import Clash.Prelude
import qualified Clash.Signal as CS
import qualified Data.List as DL
import Model.Helpers.MatVecI8E
import Model.Numeric.ParamPack (QArray2D (..))
import Model.Numeric.Types (FixedPoint)
import Test.Hspec
import qualified Prelude as P

-- Helper to create a simple quantized matrix for testing
makeSimpleQMatrix :: QArray2D 3 4
makeSimpleQMatrix =
  QArray2D
    $ (1 :> 2 :> 3 :> 4 :> Nil, 0)
    :> (5 :> 6 :> 7 :> 8 :> Nil, 0)
    :> (9 :> 10 :> 11 :> 12 :> Nil, 0)
    :> Nil

-- Helper to create a simple input vector
makeSimpleVec :: Vec 4 FixedPoint
makeSimpleVec = 1.0 :> 0.5 :> 0.25 :> 0.125 :> Nil

-- Simulate until we get a valid output, return the result
-- This waits for the handshake to complete regardless of cycle count
simulateUntilValid ::
  Int ->                        -- Max cycles to simulate
  Vec 4 FixedPoint ->          -- Input vector
  Maybe (Vec 3 FixedPoint)     -- Result (Nothing if timeout)
simulateUntilValid maxCycles vec =
  let -- Create input stream: wait for ready, then send valid input
      inputStream = (False, repeat 0) : DL.repeat (True, vec)
      inputSig = fromList inputStream

      -- Run simulation
      (outVecsSig, validOutsSig, readyOutsSig) =
        exposeClockResetEnable
          (sequentialMatVecStub makeSimpleQMatrix)
          CS.systemClockGen
          CS.resetGen
          CS.enableGen
          inputSig

      outVecs = DL.take maxCycles $ sample outVecsSig
      validOuts = DL.take maxCycles $ sample validOutsSig
      _readyOuts = DL.take maxCycles $ sample readyOutsSig

      -- Find first cycle where validOut is True
      resultsWithIndex = DL.zip3 [0 :: Int ..] validOuts outVecs
      validResults = DL.filter (\(_, valid, _) -> valid) resultsWithIndex
   in case validResults of
        [] -> Nothing  -- Timeout: no valid output within maxCycles
        ((_, _, result):_) -> Just result

spec :: Spec
spec = do
  describe "sequentialMatVecStub" $ do
    it "computes correct matrix-vector product (cycle-independent)" $ do
      let result = simulateUntilValid 10 makeSimpleVec
          
          -- Expected: [1*1 + 2*0.5 + 3*0.25 + 4*0.125,
          --            5*1 + 6*0.5 + 7*0.25 + 8*0.125,
          --            9*1 + 10*0.5 + 11*0.25 + 12*0.125]
          --         = [3.25, 10.75, 18.25]
          expected = 3.25 :> 10.75 :> 18.25 :> Nil
          tolerance = 0.01
      
      -- Check that we got a result
      result `shouldSatisfy` (  \case
                Nothing -> False
                Just _ -> True)
      
      -- Check the result is correct
      case result of
        Just outVec -> do
          let diffs = P.zipWith (\a b -> abs (a - b)) (toList outVec) (toList expected)
          all (< tolerance) diffs `shouldBe` True
        Nothing -> 
          expectationFailure "No valid output received within timeout"
