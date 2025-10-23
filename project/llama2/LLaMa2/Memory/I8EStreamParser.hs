module LLaMa2.Memory.I8EStreamParser
  ( i8eRowAssembler
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types (Mantissa, Exponent)
import LLaMa2.Numeric.Quantization (RowI8E)
import Data.Maybe (isJust)

-- State: rolling byte buffer for the current row and how many bytes are filled
type BufBytes n' = Vec (n' + 1) (BitVector 8)

-- Convert one byte to Mantissa (Signed 8)
toMant :: BitVector 8 -> Mantissa
toMant = bitCoerce

-- Convert one byte to Exponent (Signed 7, keep low 7 bits)
toExp :: BitVector 8 -> Exponent
toExp bv = bitCoerce (slice d6 d0 bv)

-- Assemble rows out of a continuous 64-byte (512-bit) stream.
-- Each row = n mantissas + 1 exponent = (n+1) bytes.
-- Emits at most one row per cycle. Extra bytes remain buffered.
i8eRowAssembler ::
  forall dom n.
  ( HiddenClockResetEnable dom
  , KnownNat n
  , KnownNat (n + 1)           -- n must be positive (satisfied in all your models)
  ) =>
  Signal dom (BitVector 512)  ->  -- ^ beat data (64 bytes)
  Signal dom Bool              ->  -- ^ beat valid
  ( Signal dom (RowI8E n)      -- ^ assembled row
  , Signal dom Bool )             -- ^ row valid
i8eRowAssembler beatData beatValid = (rowOutS, rowValidS)
  where
    -- Row size in bytes
    rowSzI :: Int
    rowSzI = natToNum @n + 1

    rowSz :: Unsigned 16
    rowSz = fromInteger (toInteger rowSzI)

    initBuf :: BufBytes n
    initBuf = repeat 0

    -- Register state
    bufS   :: Signal dom (BufBytes n)
    cntS   :: Signal dom (Unsigned 16)
    bufS = register initBuf bufNext
    cntS = register 0       cntNext

    -- Unpack beat into 64 bytes
    bytes :: Signal dom (Vec 64 (BitVector 8))
    bytes = bitCoerce <$> beatData

    -- One-cycle, at-most-one-row emission combinational step
    -- We fold the 64 incoming bytes, filling the buffer; once we reach rowSz,
    -- we produce a row and stop consuming extra bytes this cycle.
    stepOnce ::
         (BufBytes n, Unsigned 16, Maybe (RowI8E n))
      -> BitVector 8
      -> (BufBytes n, Unsigned 16, Maybe (RowI8E n))
    stepOnce st@(_, _, Just _) _ = st  -- already produced one row; ignore rest this cycle
    stepOnce (buf, cnt, Nothing) b =
      let -- write byte 'b' at position 'cnt'
          ix :: Index (n + 1)
          ix = fromIntegral cnt
          buf' = replace ix b buf
          cnt' = cnt + 1
      in if cnt' == rowSz
            then
             let mantBytes = init buf'      -- Vec (n+1) a -> Vec n a
                 expByte   = last buf'      -- gets the (n+1)th element
                 mant      = map toMant mantBytes
                 expo      = toExp expByte
             in (repeat 0, 0, Just (mant, expo))
           else
             (buf', cnt', Nothing)

    -- Apply step over a whole beat if valid
    (bufNext, cntNext, rowOutM) =
      unbundle $
        mux beatValid
          (foldl stepOnce <$> bundle (bufS, cntS, pure Nothing) <*> bytes)
          (bundle (bufS, cntS, pure Nothing))

    rowValidS = isJust <$> rowOutM
    rowOutS   = fromJustX <$> rowOutM
