module LLaMa2.Layers.Attention.MultiHeadAttention.Internal (
  applyRotaryPositionEncoding
  , applyRotation
  , computeHeadQSeq  -- NEW: sequential version
  , computeHeadKVSeq -- NEW: sequential version
) where

import Clash.Prelude
import LLaMa2.Config
    ( ModelDimension,
      HeadDimension,
      RotaryPositionalEmbeddingDimension,
      SequenceLength  )
import LLaMa2.Numeric.Types ( FixedPoint )
import LLaMa2.Layers.Components.Quantized
    ( SingleHeadComponentQ(..) )
import LLaMa2.Layers.Components.RotaryQ (RotaryEncodingComponentF (..))
import LLaMa2.Helpers.MatVecI8E (matrixMultiplier)

computeHeadQSeq
  :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool              -- ^ validIn
  -> Signal dom Bool              -- ^ readyIn (downstream ready)
  -> SingleHeadComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool            -- ^ validOut
     , Signal dom Bool            -- ^ readyOut (can accept input)
     )
computeHeadQSeq validIn readyIn headComp stepCountSig xHatSig =
  (qRoOut, validOut, readyOut)
  where
    -- Matrix multiply with handshaking
    (qOut, qValidOut, qReadyOut) =
      matrixMultiplier validIn (pure True) (wqHeadQ headComp) xHatSig

    -- Apply rotary encoding (combinational, but gated by valid)
    qRoOut = (applyRotation (rotaryQ headComp) <$> stepCountSig) <*> qOut

    -- Pass through handshaking signals
    validOut = qValidOut
    readyOut = qReadyOut

-- NEW: Sequential KV projection with handshaking
computeHeadKVSeq
  :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool              -- ^ validIn
  -> Signal dom Bool              -- ^ readyIn (downstream ready)
  -> SingleHeadComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool            -- ^ validOut
     , Signal dom Bool            -- ^ readyOut (can accept input)
     )
computeHeadKVSeq validIn readyIn headComp stepCountSig xHatSig =
  (kRoOut, vOut, validOut, readyOut)
  where
    -- K matrix multiply
    (kOut, kValidOut, kReadyOut) =
      matrixMultiplier validIn (pure True) (wkHeadQ headComp) xHatSig

    -- V matrix multiply (runs in parallel with K)
    (vOut, vValidOut, vReadyOut) =
      matrixMultiplier validIn (pure True) (wvHeadQ headComp) xHatSig

    -- Apply rotary to K
    kRoOut = (applyRotation (rotaryQ headComp) <$> stepCountSig) <*> kOut

    -- Both K and V must be valid (should happen simultaneously since same input)
    validOut = kValidOut .&&. vValidOut

    -- Can accept input only when both multipliers are ready
    readyOut = kReadyOut .&&. vReadyOut

-- Helper: safely destructure a Vec 2
vec2ToPair :: NFDataX a => Vec 2 a -> (a, a)
vec2ToPair (x :> y :> Nil) = (x, y)
vec2ToPair _   = deepErrorX "Impossible: Vec 2 had wrong shape"

applyRotaryPositionEncoding
  :: Vec HeadDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec RotaryPositionalEmbeddingDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotaryPositionEncoding inputVec cosVecF sinVecF =
  concat (imap rotatePair (unconcat d2 inputVec))
 where
  rotatePair :: Index RotaryPositionalEmbeddingDimension
             -> Vec 2 FixedPoint
             -> Vec 2 FixedPoint
  rotatePair i v =
    let (realC, imagC) = vec2ToPair v
        c  = cosVecF !! i
        s  = sinVecF !! i
        r  = realC * c - imagC * s
        im = realC * s + imagC * c
    in  r :> im :> Nil

applyRotation
  :: RotaryEncodingComponentF
  -> Index SequenceLength
  -> Vec HeadDimension FixedPoint
  -> Vec HeadDimension FixedPoint
applyRotation rot step tokenVec =
  let cosF = freqCosF rot !! step
      sinF = freqSinF rot !! step
  in  applyRotaryPositionEncoding tokenVec cosF sinF
