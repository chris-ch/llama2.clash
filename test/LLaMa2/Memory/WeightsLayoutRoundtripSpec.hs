module LLaMa2.Memory.WeightsLayoutRoundtripSpec (spec) where

import Clash.Prelude
import Test.Hspec

import LLaMa2.Numeric.Quantization (RowI8E(..), MatI8E)
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import qualified Simulation.DRAMBackedAxiSlave as DRAM
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified Prelude as P
import Clash.Sized.Vector (unsafeFromList)
import qualified Simulation.ParamsPlaceholder as PARAM
import qualified Simulation.Parameters as PARAM
import LLaMa2.Types.ModelConfig (HeadDimension, ModelDimension)
import Control.Monad (forM_)

-- Small dimension keeps the test fast
type Dim = 128
type NumWords = Layout.WordsPerRow Dim
-- NumWords = 3 for Dim=128 with the current implementation

type TestDRAMDepth = 65536
type WordData = BitVector 512

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
        packedWordsList = Layout.multiWordRowPacker row

        -- Convert the list to a static Vec.
        -- For NumWords=3 this is safe and total.
        packedWords :: Vec NumWords (BitVector 512)
        packedWords = case packedWordsList of
          [w0, w1, w2] -> w0 :> w1 :> w2 :> Nil
          _            -> error "multiWordRowPacker returned unexpected number of words"

        row' = Layout.multiWordRowParser packedWords

    row' `shouldBe` row

  it "round-trips RowI8E 4096" $ do
    let mans = iterateI (+1) (0 :: Mantissa) :: Vec 4096 Mantissa
        expon = (-8) :: Exponent
        row   = RowI8E mans expon
        packed = Layout.multiWordRowPacker row
        vec65 :: Vec (Layout.WordsPerRow 4096) (BitVector 512)
        vec65 = case packed of
                  -- Expect 65 words for 4096 in the NEW layout
                  _ | P.length packed == 65 -> unsafeFromList packed
                    | otherwise -> error "expected 65 words"
        row' = Layout.multiWordRowParser vec65
    row' `shouldBe` row

  it "Q head 0 rows [0..3] round-trip exactly" $ do
    let params = PARAM.decoderConst

        -- Build DRAM image
        dramVec :: Vec TestDRAMDepth WordData
        dramVec = DRAM.buildMemoryFromParams @TestDRAMDepth params

        -- Hardcoded Q matrix head 0 (reference)
        layer0   = head (PARAM.modelLayers params)
        mha0     = PARAM.multiHeadAttention layer0
        qHead0   :: MatI8E HeadDimension ModelDimension
        qHead0   = PARAM.wqHeadQ (head (PARAM.headsQ mha0))

        wordsPerRow = Layout.wordsPerRowVal @ModelDimension
        strideBytes = wordsPerRow * 64

        -- Helper to fetch/parse one row directly from the image
        fetchRow :: Index HeadDimension -> RowI8E ModelDimension
        fetchRow ri =
          let addrBytes :: Unsigned 32
              addrBytes = Layout.rowAddressCalculator Layout.QMatrix 0 0 ri
              baseWord  = fromIntegral (addrBytes `shiftR` 6) :: Int
              slice'     = map (\k -> dramVec !! (snatToNum (SNat @0) + toInteger (baseWord + k)))
                              (iterateI (+1) 0 :: Vec (Layout.WordsPerRow ModelDimension) Int)
          in Layout.multiWordRowParser slice'

        -- Compare rows 0..3
        go i =
          let ri       = fromInteger (toInteger i) :: Index HeadDimension
              hcRow    = qHead0 !! ri
              dramRow  = fetchRow ri
          in (hcRow, dramRow)

        results = P.map go [(0::Int)..3]

    -- All rows must match exactly
    forM_ results $ \(hcRow, dramRow) -> dramRow `shouldBe` hcRow

    -- Basic stride sanity between row0 and row1 (bytes)
    let addr0 = Layout.rowAddressCalculator Layout.QMatrix 0 0 0
        addr1 = Layout.rowAddressCalculator Layout.QMatrix 0 0 1
        delta = fromIntegral (addr1 - addr0) :: Int
    delta `shouldBe` strideBytes
