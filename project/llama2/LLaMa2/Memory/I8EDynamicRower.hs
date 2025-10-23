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

-- Compute per-segment row length (bytes)
rowLenBytes :: SNat md -> SNat hd -> SNat hid -> LayerSeg -> Unsigned 16
rowLenBytes mdNat hdNat hidNat s = case s of
  SegRmsAtt -> fromInteger (snatToNum mdNat + 1)
  SegQ      -> fromInteger (snatToNum mdNat + 1)
  SegK      -> fromInteger (snatToNum mdNat + 1)
  SegV      -> fromInteger (snatToNum mdNat + 1)
  SegWO     -> fromInteger (snatToNum hdNat + 1)
  SegRmsFfn -> fromInteger (snatToNum mdNat + 1)
  SegW1     -> fromInteger (snatToNum hidNat + 1)
  SegW2     -> fromInteger (snatToNum hidNat + 1)
  SegW3     -> fromInteger (snatToNum hidNat + 1)

isMD :: LayerSeg -> Bool
isMD s = case s of
  SegRmsAtt -> True
  SegQ      -> True
  SegK      -> True
  SegV      -> True
  SegRmsFfn -> True
  _         -> False

-- Simple circular FIFO for bytes; depth must be multiple of 64
data FifoState = FifoState
  { rdPtr :: Unsigned 12    -- supports up to 4096 depth (2^12)
  , wrPtr :: Unsigned 12
  , used  :: Unsigned 13    -- up to 4096
  } deriving (Generic, NFDataX, Show, Eq)

-- FIFO storage
type Store = Vec 4096 Byte

dynamicRower ::
  forall dom md hd hid.
  ( HiddenClockResetEnable dom
  , KnownNat md
  ) =>
  SNat md -> SNat hd -> SNat hid ->
  Signal dom (BitVector 512) ->  -- ^ beat data
  Signal dom Bool             ->  -- ^ beat valid
  Signal dom LayerSeg         ->  -- ^ current segment (from address generator)
  ( Signal dom (RowI8E md)    -- ^ mdRowOut: only for MD segments (Q/K/V/rms*)
  , Signal dom Bool           -- ^ mdRowValid
  , Signal dom Bool           -- ^ rowDoneExt (all segments)
  , Signal dom Bool           -- ^ sinkReady (AXI may accept a beat)
  )
dynamicRower _md _hd _hid beatData beatValid segS =
  (mdRowOutS, mdRowValidS, rowDoneExtS, sinkReadyS)
 where
  -- Parameters
  depth :: Unsigned 13
  depth = 4096

  storeS :: Signal dom Store
  storeS = register (repeat 0) storeNext

  stS :: Signal dom FifoState
  stS = register (FifoState 0 0 0) stNext

  -- Unpack incoming beat into bytes
  beatBytes = bitCoerce <$> beatData :: Signal dom (Vec 64 Byte)

  canPush64 s = (used s + 64) <= depth
  sinkReadyS = canPush64 <$> stS

  -- Decide pop length based on segment
  rowLenBytesS :: SNat md -> SNat hd -> SNat hid -> Signal dom LayerSeg -> Signal dom (Unsigned 16)
  rowLenBytesS mdNat hdNat hidNat = fmap (rowLenBytes mdNat hdNat hidNat)
  rowLenS = rowLenBytesS _md _hd _hid segS

  -- Coercion helpers
  toUsed13 :: Unsigned 16 -> Unsigned 13
  toUsed13 = resize

  toPtr12 :: Unsigned 16 -> Unsigned 12
  toPtr12 = resize

  canPop s rl = used s >= toUsed13 rl

  -- Pop decision: one row per cycle max
  willPopS = canPop <$> stS <*> rowLenS

  -- Write 64 bytes when beatValid && sinkReady
  pushNowS = beatValid .&&. sinkReadyS

  -- Next state/store
  stNext = nextState <$> stS <*> pushNowS <*> willPopS <*> rowLenS
  storeNext = nextStore <$> storeS <*> stS <*> pushNowS <*> beatBytes

  nextState FifoState{..} pushNow popNow rl =
    let used'  = used + (if pushNow then 64 else 0) - toUsed13 rl * (if popNow then 1 else 0)
        wrPtr' = wrPtr + (if pushNow then 64 else 0)
        rdPtr' = rdPtr + toPtr12 rl * (if popNow then 1 else 0)
    in FifoState rdPtr' wrPtr' used'

  write64 :: Unsigned 12 -> Store -> Vec 64 Byte -> Store
  write64 base = ifoldl step
    where
      step :: Store -> Index 64 -> Byte -> Store
      step acc i b = replace (base + fromIntegral (fromEnum i)) b acc

  readRow :: forall n. KnownNat n => Store -> Unsigned 12 -> (Vec n Byte, Byte)
  readRow st base =
    let n  = fromInteger (natToNum @n) :: Int
        -- read n mantissas
        m  = map (\(i :: Index n) -> st !! (base + fromIntegral (fromEnum i))) indicesI
        -- exponent at offset n
        e  = st !! (base + fromIntegral n)
    in (m, e)

  nextStore :: Store -> FifoState -> Bool -> Vec 64 Byte -> Store
  nextStore st FifoState{..} pushNow xs = if pushNow then write64 wrPtr st xs else st

  -- Output logic (combinational from current store/state)
  mdRowOutComb :: Store -> FifoState -> LayerSeg -> Unsigned 16 -> (RowI8E md, Bool, Bool)
  mdRowOutComb st FifoState{..} seg rl =
    let popNow = used >= toUsed13 rl
        (mantB, expB) = readRow @md st rdPtr
        mdRow = (map toMant mantB, toExp expB)
        mdValid = popNow && isMD seg
        rowDone = popNow
    in (mdRow, mdValid, rowDone)

  (mdRowOutS, mdRowValidS, rowDoneExtS) =
    unbundle (mdRowOutComb <$> storeS <*> stS <*> segS <*> rowLenS)
