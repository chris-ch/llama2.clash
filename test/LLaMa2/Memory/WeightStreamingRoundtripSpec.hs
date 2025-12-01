module LLaMa2.Memory.WeightStreamingRoundtripSpec (spec) where

import Clash.Prelude
import Test.Hspec

import LLaMa2.Numeric.Quantization (RowI8E(..))
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified LLaMa2.Memory.WeightStreaming as STREAM

-- Small dimension keeps the test fast
type Dim = 128
type NumWords = STREAM.WordsPerRow Dim
-- NumWords = 3 for Dim=128 with the current implementation

mkRow :: Vec Dim Mantissa -> Exponent -> RowI8E Dim
mkRow ms e = RowI8E { rowMantissas = ms, rowExponent = e }

spec :: Spec
spec = describe "RowI8E pack/parse round-trip" $ do
  it "round-trips mantissas and exponent exactly" $ do
    let mans = iterateI (+1) (0 :: Mantissa)       -- 0,1,2,...,127,0,...
        expon  = -5 :: Exponent
        row  = mkRow mans expon

        -- Pack into DRAM words (produces a list)
        packedWordsList :: [BitVector 512]
        packedWordsList = DRAM.packRowMultiWord row

        -- Convert the list to a static Vec.
        -- For NumWords=3 this is safe and total.
        packedWords :: Vec NumWords (BitVector 512)
        packedWords = case packedWordsList of
          [w0, w1, w2] -> w0 :> w1 :> w2 :> Nil
          _            -> error "packRowMultiWord returned unexpected number of words"

        row' = STREAM.multiWordRowParser packedWords

    row' `shouldBe` row
