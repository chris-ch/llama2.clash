{-# LANGUAGE AllowAmbiguousTypes #-}
module LLaMa2.Memory.I8EDynamicRower
  ( dynamicRower
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import LLaMa2.Numeric.Quantization (RowI8E)
import LLaMa2.Memory.LayerAddressing (LayerSeg(..))

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

isMD :: LayerSeg -> Bool
isMD s = case s of
  SegRmsAtt -> True
  SegQ      -> True
  SegK      -> True
  SegV      -> True
  SegRmsFfn -> True
  _         -> False

isHD :: LayerSeg -> Bool
isHD s  = s == SegWO

isHID :: LayerSeg -> Bool
isHID s = s == SegW2

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

-- New: full triple-output rower
dynamicRower3 ::
  forall dom md hd hid.
  ( HiddenClockResetEnable dom
  , KnownNat md, KnownNat hd, KnownNat hid
  ) =>
  SNat md -> SNat hd -> SNat hid ->
  Signal dom (BitVector 512) ->  -- ^ beat data
  Signal dom Bool             ->  -- ^ beat valid
  Signal dom LayerSeg         ->  -- ^ current segment
  ( Signal dom (RowI8E md)    -- ^ mdRowOut (only for MD segments)
  , Signal dom Bool   -- md row + valid (rmsAtt/Q/K/V/rmsFfn)
  , Signal dom (RowI8E hd)
  ,  Signal dom Bool   -- hd row + valid (WO)
  , Signal dom (RowI8E hid)
  , Signal dom Bool   -- hid row + valid (W2)
  , Signal dom Bool           -- ^ rowDoneExt (all segments)
  , Signal dom Bool           -- ^ sinkReady (can accept a beat)
  )
dynamicRower3 md hd hid beatData beatValid segS =
  (mdRowOutS, mdRowValidS, hdRowOutS, hdRowValidS, hidRowOutS, hidRowValidS, rowDoneExtS, sinkReadyS)
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
  pushNowS   = beatValid .&&. sinkReadyS

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
  readRowN :: forall n. KnownNat n => Store -> Unsigned 12 -> (Vec n Byte, Byte)
  readRowN st base =
    let n = (natToNum @n) :: Int
        mant = map (\(i :: Index n) -> loadByte st (plusU12 base (resize (fromIntegral (fromEnum i) :: Unsigned 12)))) indicesI
        expB = loadByte st (plusU12 base (resize (fromIntegral n :: Unsigned 12)))
    in (mant, expB)

  nextStore st FifoState{..} pushNow xs =
    if pushNow then write64 st wrPtr xs else st


  -- Outputs (combinational from current state)
  mdComb :: Store -> FifoState -> LayerSeg -> Unsigned 13 -> (RowI8E md,RowI8E hd,RowI8E hid, Bool, Bool, Bool)
  mdComb st FifoState{..} seg rl =
    let popNow = used >= rl
        (mMdB, eMdB)   = readRowN @md  st rdPtr
        (mHdB, eHdB)   = readRowN @hd  st rdPtr
        (mHidB, eHidB) = readRowN @hid st rdPtr
        mdRow  = (map toMant mMdB, toExp eMdB)
        hdRow  = (map toMant mHdB, toExp eHdB)
        hidRow = (map toMant mHidB, toExp eHidB)
        mdV  = popNow && isMD  seg
        hdV  = popNow && isHD  seg
        hidV = popNow && isHID seg
    in (mdRow, hdRow, hidRow, mdV, hdV, hidV)

  (mdRowOutS, hdRowOutS, hidRowOutS, mdRowValidS, hdRowValidS, hidRowValidS) =
    let tup = mdComb <$> storeS <*> stS <*> segS <*> rowLenS
      in (  (\(a,_,_,_,_,_)->a) <$> tup
        ,  (\(_,b,_,_,_,_)->b) <$> tup
        ,  (\(_,_,c,_,_,_)->c) <$> tup
        ,  (\(_,_,_,d,_,_)->d) <$> tup
        ,  (\(_,_,_,_,e,_)->e) <$> tup
        ,  (\(_,_,_,_,_,f)->f) <$> tup )

  rowDoneExtS = willPopS

-- Backward-compatible wrapper (MD only)
dynamicRower ::
  forall dom md hd hid.
  ( HiddenClockResetEnable dom
  , KnownNat md, KnownNat hd, KnownNat hid
  ) =>
  SNat md -> SNat hd -> SNat hid ->
  Signal dom (BitVector 512) ->
  Signal dom Bool ->
  Signal dom LayerSeg ->
  ( Signal dom (RowI8E md), Signal dom Bool, Signal dom Bool, Signal dom Bool )
dynamicRower md hd hid d v s =
  let (m, mv, _h, _hv, _i, _iv, done, rdy) = dynamicRower3 md hd hid d v s
  in (m, mv, done, rdy)
