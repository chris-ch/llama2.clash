module LLaMa2.Memory.LayerAddressing
  ( LayerSeg(..)
  , LayerAddress(..)
  , layerAddressGenerator
  , rowsInSeg
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( ModelDimension
  , HeadDimension
  , HiddenDimension
  , NumQueryHeads
  , NumKeyValueHeads
  )

-- Segment order MUST match on-disk layout:
-- rmsAtt -> Q -> K -> V -> WO -> rmsFfn -> W1 -> W2 -> W3
data LayerSeg
  = SegRmsAtt
  | SegQ | SegK | SegV
  | SegWO
  | SegRmsFfn
  | SegW1 | SegW2 | SegW3
  deriving (Generic, NFDataX, Show, Eq, Enum, Bounded)

-- Address visible to buffer logic.
-- rowIx is only meaningful for Q/K/V (HeadDimension rows). For other segs it's 0.
data LayerAddress = LayerAddress
  { seg    :: LayerSeg
  , headIx :: Unsigned 8
  , rowIx  :: Index HeadDimension
  } deriving (Generic, NFDataX, Show, Eq)

-- Row counts per segment (rows of (rowWidth+1) bytes)
rowsInSeg :: LayerSeg -> Unsigned 16
rowsInSeg s = case s of
  SegRmsAtt -> 1
  SegQ      -> fromInteger $ natToNum @NumQueryHeads     * natToNum @HeadDimension
  SegK      -> fromInteger $ natToNum @NumKeyValueHeads  * natToNum @HeadDimension
  SegV      -> fromInteger $ natToNum @NumKeyValueHeads  * natToNum @HeadDimension
  SegWO     -> fromInteger $ natToNum @NumQueryHeads     * natToNum @ModelDimension
  SegRmsFfn -> 1
  SegW1     -> fromInteger $ natToNum @HiddenDimension
  SegW2     -> fromInteger $ natToNum @ModelDimension
  SegW3     -> fromInteger $ natToNum @HiddenDimension

-- Internal per-segment limits
rowLimitFor :: LayerSeg -> Unsigned 16
rowLimitFor s = case s of
  SegRmsAtt -> 1
  SegQ      -> fromInteger $ natToNum @HeadDimension
  SegK      -> fromInteger $ natToNum @HeadDimension
  SegV      -> fromInteger $ natToNum @HeadDimension
  SegWO     -> fromInteger $ natToNum @ModelDimension
  SegRmsFfn -> 1
  SegW1     -> fromInteger $ natToNum @HiddenDimension
  SegW2     -> fromInteger $ natToNum @ModelDimension
  SegW3     -> fromInteger $ natToNum @HiddenDimension

headLimitFor :: LayerSeg -> Unsigned 16
headLimitFor s = case s of
  SegQ  -> fromInteger $ natToNum @NumQueryHeads
  SegWO -> fromInteger $ natToNum @NumQueryHeads
  SegK  -> fromInteger $ natToNum @NumKeyValueHeads
  SegV  -> fromInteger $ natToNum @NumKeyValueHeads
  _     -> 1

nextSeg :: LayerSeg -> LayerSeg
nextSeg SegRmsAtt = SegQ
nextSeg SegQ      = SegK
nextSeg SegK      = SegV
nextSeg SegV      = SegWO
nextSeg SegWO     = SegRmsFfn
nextSeg SegRmsFfn = SegW1
nextSeg SegW1     = SegW2
nextSeg SegW2     = SegW3
nextSeg SegW3     = SegRmsAtt

-- Address generator driven by completed rows (rowValid). Reset on layer change.
layerAddressGenerator :: forall dom .
  HiddenClockResetEnable dom =>
  Signal dom Bool                ->  -- ^ rowValid (assembled row)
  Signal dom Bool                ->  -- ^ reset (layer change / firstCycle)
  ( Signal dom LayerAddress
  , Signal dom Bool              )   -- ^ allDone (pulse on end of W3)
layerAddressGenerator rowValid reset = (addrOut, donePulse)
  where
    -- State
    segReg  :: Signal dom LayerSeg
    segReg  = register SegRmsAtt segNextS

    rowCntS :: Signal dom (Unsigned 16)
    rowCntS = register 0 rowNextS

    headCntS :: Signal dom (Unsigned 16)
    headCntS = register 0 headNextS

    -- Current limits
    rowLimS  = fmap rowLimitFor  segReg
    headLimS = fmap headLimitFor segReg

    -- Helpers
    step     = rowValid
    rowWrapS = liftA3 (\rv rc rl -> rv && (rc + 1 == rl)) step rowCntS rowLimS
    headWrapS= liftA3 (\rw hc hl -> rw && (hc + 1 == hl)) rowWrapS headCntS headLimS

    -- Next counters
    rowNextS =
      mux reset 0 $
      mux step (mux rowWrapS 0 (rowCntS + 1)) rowCntS

    headNextS =
      mux reset 0 $
      mux rowWrapS (mux headWrapS 0 (headCntS + 1)) headCntS

    segNextS =
      mux reset (pure SegRmsAtt) $
      mux headWrapS (nextSeg <$> segReg) segReg

    -- Done pulse at the exact end of the W3 segment
    doneNow  = (segReg .==. pure SegW3) .&&. headWrapS
    donePulse= register False (mux reset (pure False) doneNow)

    -- Public address
    headIxOut :: Signal dom (Unsigned 8)
    headIxOut = resize <$> headCntS

    -- Only meaningful for Q/K/V; otherwise 0
    rowIxOut :: Signal dom (Index HeadDimension)
    rowIxOut =
      let clampToHD :: Unsigned 16 -> Index HeadDimension
          clampToHD u =
            let m = fromIntegral (maxBound :: Index HeadDimension) :: Int
                v = fromIntegral u :: Int
            in toEnum (min m v)
          in mux ((segReg .==. pure SegQ) .||. (segReg .==. pure SegK) .||. (segReg .==. pure SegV))
                 (clampToHD <$> rowCntS)
                 (pure 0)

    addrOut = LayerAddress <$> segReg <*> headIxOut <*> rowIxOut
