module LLaMa2.Layer.Attention.WeightBuffer
  ( SingleHeadWeightBuffer(..)
  , QKVWeightBuffer(..)
  , qkvWeightBufferController
  , extractQWeight
  , extractKWeight
  , extractVWeight
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( ModelDimension
  , HeadDimension
  , NumQueryHeads
  , NumKeyValueHeads
  )
import LLaMa2.Numeric.Quantization (RowI8E, MatI8E)
import LLaMa2.Memory.WeightLoaderAddressing
  ( WeightAddress(..)
  , WeightMatrixType(..)
  )

-- Buffer for one head: holds its Q/K/V matrices (per-row I8E)
data SingleHeadWeightBuffer = SingleHeadWeightBuffer
  { wqBuf      :: MatI8E HeadDimension ModelDimension
  , wkBuf      :: MatI8E HeadDimension ModelDimension
  , wvBuf      :: MatI8E HeadDimension ModelDimension
  , loadedRows :: Unsigned 16
  } deriving (Generic, NFDataX)

emptyHeadBuffer :: SingleHeadWeightBuffer
emptyHeadBuffer = SingleHeadWeightBuffer
  { wqBuf      = repeat (repeat 0, 0)
  , wkBuf      = repeat (repeat 0, 0)
  , wvBuf      = repeat (repeat 0, 0)
  , loadedRows = 0
  }

-- Full Q/K/V buffer across all heads
data QKVWeightBuffer = QKVWeightBuffer
  { qHeadBuffers  :: Vec NumQueryHeads     SingleHeadWeightBuffer
  , kvHeadBuffers :: Vec NumKeyValueHeads  SingleHeadWeightBuffer
  , fullyLoaded   :: Bool
  } deriving (Generic, NFDataX)

emptyQKVBuffer :: QKVWeightBuffer
emptyQKVBuffer = QKVWeightBuffer
  { qHeadBuffers  = repeat emptyHeadBuffer
  , kvHeadBuffers = repeat emptyHeadBuffer
  , fullyLoaded   = False
  }

-- Safe cast from Unsigned 8 to bounded Index m; Nothing if out-of-range.
toIndexMaybe :: forall m. KnownNat m => Unsigned 8 -> Maybe (Index m)
toIndexMaybe u =
  let limI = natToNum @m :: Integer
      vI   = toInteger u
  in if vI < limI then Just (toEnum (fromInteger vI)) else Nothing

-- Pure update: place one row into the correct head/matrix/row.
-- fullyLoaded is not decided here; the controller latches it.
updateQKVBufferOnce ::
     WeightAddress
  -> RowI8E ModelDimension
  -> QKVWeightBuffer
  -> QKVWeightBuffer
updateQKVBufferOnce WeightAddress{..} row buf =
  case matrixType of
    QMatrix ->
      case toIndexMaybe @NumQueryHeads headIndex of
        Nothing -> buf
        Just hIx ->
          let hb  = qHeadBuffers buf !! hIx
              hb' = hb { wqBuf = replace rowIndex row (wqBuf hb)
                       , loadedRows = loadedRows hb + 1 }
          in buf { qHeadBuffers = replace hIx hb' (qHeadBuffers buf) }

    KMatrix ->
      case toIndexMaybe @NumKeyValueHeads headIndex of
        Nothing -> buf
        Just hIx ->
          let hb  = kvHeadBuffers buf !! hIx
              hb' = hb { wkBuf = replace rowIndex row (wkBuf hb)
                       , loadedRows = loadedRows hb + 1 }
          in buf { kvHeadBuffers = replace hIx hb' (kvHeadBuffers buf) }

    VMatrix ->
      case toIndexMaybe @NumKeyValueHeads headIndex of
        Nothing -> buf
        Just hIx ->
          let hb  = kvHeadBuffers buf !! hIx
              hb' = hb { wvBuf = replace rowIndex row (wvBuf hb)
                       , loadedRows = loadedRows hb + 1 }
          in buf { kvHeadBuffers = replace hIx hb' (kvHeadBuffers buf) }

-- Stateful controller: accumulates rows when streamValid; clears on reset.
-- fullyLoaded latches high when allRowsReceived becomes True, until reset.
qkvWeightBufferController ::
  HiddenClockResetEnable dom
  => Signal dom Bool                       -- ^ streamValid
  -> Signal dom WeightAddress              -- ^ address (from Step 1)
  -> Signal dom (RowI8E ModelDimension)    -- ^ weight row (parsed)
  -> Signal dom Bool                       -- ^ allRowsReceived (qkvLoadDone pulse)
  -> Signal dom Bool                       -- ^ reset (new layer trigger)
  -> Signal dom QKVWeightBuffer            -- ^ buffered weights
qkvWeightBufferController streamValid addr row allDone reset = bufS
  where
    stepEn = streamValid .||. reset

    bufNext =
      mux reset
        (pure emptyQKVBuffer)
        (apply <$> streamValid <*> addr <*> row <*> allDone <*> bufS)

    apply en a r done prev =
      let updated   = if en then updateQKVBufferOnce a r prev else prev
          loadedLat = fullyLoaded prev || done
      in updated { fullyLoaded = loadedLat }

    bufS = regEn emptyQKVBuffer stepEn bufNext

-- Extraction helpers
extractQWeight :: QKVWeightBuffer -> Index NumQueryHeads     -> MatI8E HeadDimension ModelDimension
extractQWeight buf hIx = wqBuf (qHeadBuffers buf !! hIx)

extractKWeight :: QKVWeightBuffer -> Index NumKeyValueHeads  -> MatI8E HeadDimension ModelDimension
extractKWeight buf hIx = wkBuf (kvHeadBuffers buf !! hIx)

extractVWeight :: QKVWeightBuffer -> Index NumKeyValueHeads  -> MatI8E HeadDimension ModelDimension
extractVWeight buf hIx = wvBuf (kvHeadBuffers buf !! hIx)
