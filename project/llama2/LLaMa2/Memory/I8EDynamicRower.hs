{-# LANGUAGE AllowAmbiguousTypes #-}
module LLaMa2.Memory.I8EDynamicRower
  ( dynamicRower ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Memory.WeightLoaderAddressingExtended (LayerSeg(..))

type Byte = BitVector 8

toMant :: Byte -> Mantissa
toMant = bitCoerce

toExp :: Byte -> Exponent
toExp b = bitCoerce (slice d6 d0 b)

-- Per-segment row length in BYTES. Must match on-disk layout:
-- - MD-width rows: rmsAtt, Q, K, V, rmsFfn → ModelDimension + 1
-- - WO rows       : WO                     → HeadDimension + 1
-- - FFN rows      : W1, W3                 → ModelDimension + 1
-- - FFN rows      : W2                     → HiddenDimension + 1
rowLenBytes :: forall md hd hid . (KnownNat md, KnownNat hd, KnownNat hid)
            => LayerSeg -> Unsigned 13
rowLenBytes s = case s of
  SegRmsAtt ->  natToNum @md + 1
  SegQ      ->  natToNum @md + 1
  SegK      ->  natToNum @md + 1
  SegV      ->  natToNum @md + 1
  SegWO     ->  natToNum @hd + 1
  SegRmsFfn ->  natToNum @md + 1
  SegW1     ->  natToNum @md + 1
  SegW2     ->  natToNum @hid + 1
  SegW3     ->  natToNum @md + 1

-- MD segments are the only ones for which we emit a typed RowI8E md
isMD :: LayerSeg -> Bool
isMD s = case s of
  SegRmsAtt -> True
  SegQ      -> True
  SegK      -> True
  SegV      -> True
  SegRmsFfn -> True
  _         -> False

-- Simple circular FIFO for bytes; depth = 4096 (must be multiple of 64)
data FifoState = FifoState
  { rdPtr :: Unsigned 12   -- 0..4095
  , wrPtr :: Unsigned 12
  , used  :: Unsigned 13   -- 0..4096
  } deriving (Generic, NFDataX, Show, Eq)

-- Index helpers: wrap 12-bit address to Index 4096
toIdx4096 :: Unsigned 12 -> Index 4096
toIdx4096 u = toEnum (fromIntegral u)

plusU12 :: Unsigned 12 -> Unsigned 12 -> Unsigned 12
plusU12 a b = resize (a + b)  -- wraps mod 4096 automatically

type Store = Vec 4096 Byte

dynamicRower ::
  forall dom md hd hid.
  ( HiddenClockResetEnable dom
  , KnownNat md, KnownNat hd, KnownNat hid
  ) =>
  SNat md -> SNat hd -> SNat hid ->
  Signal dom (BitVector 512) ->  -- ^ beat data
  Signal dom Bool             ->  -- ^ beat valid
  Signal dom LayerSeg         ->  -- ^ current segment
  ( Signal dom (RowI8E md)    -- ^ mdRowOut (only for MD segments)
  , Signal dom Bool           -- ^ mdRowValid
  , Signal dom Bool           -- ^ rowDoneExt (all segments)
  , Signal dom Bool           -- ^ sinkReady (can accept a beat)
  )
dynamicRower _md _hd _hid beatData beatValid segS =
  (mdRowOutS, mdRowValidS, rowDoneExtS, sinkReadyS)
 where
  -- FIFO storage
  storeS :: Signal dom Store
  storeS = register (repeat 0) storeNext

  stS :: Signal dom FifoState
  stS = register (FifoState 0 0 0) stNext

  -- Unpack incoming beat into 64 bytes
  beatBytes = bitCoerce <$> beatData :: Signal dom (Vec 64 Byte)

  -- Ready to push another 64 bytes?
  canPush64 s = (used s + 64) <= 4096
  sinkReadyS = canPush64 <$> stS
  pushNowS   = beatValid .&&.sinkReadyS

  -- Per-segment row length
  rowLenS = rowLenBytes @md @hd @hid <$> segS

  -- Can pop one full row this cycle?
  canPop s rl = used s >= rl
  willPopS = canPop <$> stS <*> rowLenS

  -- Next FIFO state
  stNext = nextState <$> stS <*> pushNowS <*> willPopS <*> rowLenS
  storeNext = nextStore <$> storeS <*> stS <*> pushNowS <*> beatBytes

  nextState FifoState{..} pushNow popNow rl =
    let used'  = used + (if pushNow then 64 else 0) - (if popNow then rl else 0)
        wrPtr' = plusU12 wrPtr (if pushNow then 64 else 0)
        rdPtr' = plusU12 rdPtr (if popNow then resize rl else 0)
    in FifoState rdPtr' wrPtr' used'

  -- Store writers/readers with wrap
  write64 :: Store -> Unsigned 12 -> Vec 64 Byte -> Store
  write64 st base xs =
    let doOne acc (i :: Index 64) =
          let addr = plusU12 base (resize (fromIntegral (fromEnum i) :: Unsigned 12))
          in replace (toIdx4096 addr) (xs !! i) acc
    in foldl doOne st indicesI

  loadByte :: Store -> Unsigned 12 -> Byte
  loadByte st addr = st !! toIdx4096 addr

  -- Read a row of length rl bytes starting at base; first md mantissas + 1 exp for MD segments
  readRowMD :: forall n. KnownNat n => Store -> Unsigned 12 -> (Vec n Byte, Byte)
  readRowMD st base =
    let n = (natToNum @n) :: Int
        mant = map (\(i :: Index n) -> loadByte st (plusU12 base (resize (fromIntegral (fromEnum i) :: Unsigned 12)))) indicesI
        expB = loadByte st (plusU12 base (resize (fromIntegral n :: Unsigned 12)))
    in (mant, expB)

  nextStore st FifoState{..} pushNow xs =
    let st1 = if pushNow then write64 st wrPtr xs else st
        -- Pop doesn’t change stored bytes; rdPtr/used change in state only
    in st1

  -- Outputs (combinational from current state)
  mdRowOutComb :: Store -> FifoState -> LayerSeg -> Unsigned 13 -> (RowI8E md, Bool, Bool)
  mdRowOutComb st FifoState{..} seg rl =
    let popNow = used >= rl
        (mantB, expB) = readRowMD @md st rdPtr
        mdRow   = (map toMant mantB, toExp expB)
        mdValid = popNow && isMD seg
        rowDone = popNow
    in (mdRow, mdValid, rowDone)

  (mdRowOutS, mdRowValidS, rowDoneExtS) =
    unbundle (mdRowOutComb <$> storeS <*> stS <*> segS <*> rowLenS)
  