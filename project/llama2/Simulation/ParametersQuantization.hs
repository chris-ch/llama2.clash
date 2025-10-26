module Simulation.ParametersQuantization (
    quantizeMatI8E
) where
import Clash.Prelude
import LLaMa2.Types.LayerData (CArray2D (..))
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Numeric.Types (FixedPoint, Exponent, Activation, scalePow2F, clampExp, satRoundToI8)

-- Elaborate-time quantization: Float -> FixedPoint -> I8E per row.
-- Safe for synthesis because inputs are structural constants.
quantizeMatI8E
  :: ( KnownNat cols)
  => CArray2D rows cols                 -- Float params baked in the netlist
  -> MatI8E rows cols                 -- Float-free carrier for hardware
quantizeMatI8E (CArray2D rowsF) =
  let rowsFtoF = map (map realToFrac) rowsF
  in map quantizeI8E rowsFtoF

-- ===========================
-- Quantization: F --> I8E (PoT)
-- ===========================

-- Quantize a vector to Signed 8 mantissas with a shared Signed 7 exponent.
-- Nearest-PoT exponent (reduces MSE vs floor-PoT)
-- No Floating is used to find the exponent: we compare against a ROM of 2^i.
quantizeI8E :: forall n. KnownNat n => Vec n FixedPoint -> (Vec n Activation, Exponent)
quantizeI8E xs =
  let maxAbs :: FixedPoint
      maxAbs = foldl max 0 (map abs xs)
      pow2 :: Vec 64 FixedPoint
      pow2 = map (\(i :: Index 64) ->
                    let iS :: Exponent
                        iS = fromInteger (toInteger (fromEnum i) - 32)
                    in scalePow2F iS 1)
                 indicesI
      flags :: Vec 64 Bool
      flags = map (<= maxAbs) pow2
      pIdx :: Index 64
      pIdx = fst (foldl (\(best, seen) (i,b) -> if b then (i,True) else (best,seen))
                        (minBound, False)
                        (zip indicesI flags))
      pInt :: Integer
      pInt = toInteger (fromEnum pIdx) - 32
      eFloor :: Exponent
      eFloor = clampExp (fromInteger (pInt - 7))
      eCeil  :: Exponent
      eCeil  = clampExp (fromInteger (pInt - 6))
      errFor :: Exponent -> FixedPoint
      errFor e =
        let s  = scalePow2F (negate e) 1
            qf x = satRoundToI8 (round (x * s))
            rec x = fromIntegral (qf x) * scalePow2F e 1
        in sum (map (\x -> let d = x - rec x in d*d) xs)
      eBest = if errFor eFloor <= errFor eCeil then eFloor else eCeil
      k = scalePow2F (negate eBest) 1
      qElem x =
        let y  = x * k
            yr = if y >= 0 then floor (y + 0.5) else ceiling (y - 0.5) :: Integer
        in satRoundToI8 yr
  in (map qElem xs, eBest)
