{-# LANGUAGE RecordWildCards #-}
module LLaMa2.Layer.Attention.LayerWeightBuffer
  ( LayerWeightBuffer(..)
  , layerWeightBufferController
  , extractQKV
  , extractW1
  , extractW3
  , extractRmsAtt
  , extractRmsFfn
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( ModelDimension
  , HiddenDimension
  , NumQueryHeads
  , NumKeyValueHeads
  )
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import LLaMa2.Layer.Attention.WeightBuffer
  ( QKVWeightBuffer(..)
  , SingleHeadWeightBuffer(..)
  )
import LLaMa2.Memory.LayerAddressing
  ( LayerSeg(..), LayerAddress(..) )

-- Full per-layer RAM buffer (subset for now: Q/K/V + W1 + W3 + RMS vectors)
data LayerWeightBuffer = LayerWeightBuffer
  { qkv            :: QKVWeightBuffer
  , w1Buf          :: MatI8E HiddenDimension ModelDimension
  , w3Buf          :: MatI8E HiddenDimension ModelDimension
  , rmsAttRow      :: RowI8E ModelDimension
  , rmsFfnRow      :: RowI8E ModelDimension
  , w1LoadedRows   :: Unsigned 16
  , w3LoadedRows   :: Unsigned 16
  , fullyLoadedAll :: Bool
  } deriving (Generic, NFDataX)

-- Zero/empty initializers
emptySingleHead :: SingleHeadWeightBuffer
emptySingleHead = SingleHeadWeightBuffer
  { wqBuf = repeat (repeat 0, 0)
  , wkBuf = repeat (repeat 0, 0)
  , wvBuf = repeat (repeat 0, 0)
  , loadedRows = 0
  }

emptyQKV :: QKVWeightBuffer
emptyQKV = QKVWeightBuffer
  { qHeadBuffers  = repeat emptySingleHead
  , kvHeadBuffers = repeat emptySingleHead
  , fullyLoaded   = False
  }

emptyLayerBuffer :: LayerWeightBuffer
emptyLayerBuffer = LayerWeightBuffer
  { qkv            = emptyQKV
  , w1Buf          = repeat (repeat 0, 0)
  , w3Buf          = repeat (repeat 0, 0)
  , rmsAttRow      = (repeat 0, 0)
  , rmsFfnRow      = (repeat 0, 0)
  , w1LoadedRows   = 0
  , w3LoadedRows   = 0
  , fullyLoadedAll = False
  }

-- Safely turn Unsigned into Index, clamping to range
toIndexClamp :: forall n. KnownNat n => Unsigned 16 -> Index n
toIndexClamp u =
  let m = fromIntegral (maxBound :: Index n) :: Int
      v = fromIntegral u :: Int
  in toEnum (min m v)

toIndexMaybe :: forall n. KnownNat n => Unsigned 8 -> Maybe (Index n)
toIndexMaybe u =
  let lim = natToNum @n :: Integer
      v   = toInteger u
  in if v < lim then Just (toEnum (fromInteger v)) else Nothing

-- Pure update of QKV sub-buffer for one extended address row
updateQKVOnce ::
     LayerAddress
  -> RowI8E ModelDimension
  -> QKVWeightBuffer
  -> QKVWeightBuffer
updateQKVOnce LayerAddress{..} row buf =
  case seg of
    SegQ ->
      case toIndexMaybe @NumQueryHeads headIx of
        Nothing -> buf
        Just hi ->
          let hb  = qHeadBuffers buf !! hi
              hb' = hb { wqBuf = replace rowIx row (wqBuf hb)
                       , loadedRows = loadedRows hb + 1
                       }
          in buf { qHeadBuffers = replace hi hb' (qHeadBuffers buf) }
    SegK ->
      case toIndexMaybe @NumKeyValueHeads headIx of
        Nothing -> buf
        Just hi ->
          let hb  = kvHeadBuffers buf !! hi
              hb' = hb { wkBuf = replace rowIx row (wkBuf hb)
                       , loadedRows = loadedRows hb + 1
                       }
          in buf { kvHeadBuffers = replace hi hb' (kvHeadBuffers buf) }
    SegV ->
      case toIndexMaybe @NumKeyValueHeads headIx of
        Nothing -> buf
        Just hi ->
          let hb  = kvHeadBuffers buf !! hi
              hb' = hb { wvBuf = replace rowIx row (wvBuf hb)
                       , loadedRows = loadedRows hb + 1
                       }
          in buf { kvHeadBuffers = replace hi hb' (kvHeadBuffers buf) }
    _ -> buf

-- Stateful controller: update on rowValid; clear on reset; latch fullyLoadedAll on allDone pulse.
-- NOTE: For this increment we populate Q/K/V + W1 + W3 + RMS vectors. WO and W2 are ignored here.
layerWeightBufferController ::
  HiddenClockResetEnable dom =>
  Signal dom Bool                      -- ^ rowValid (assembled row)
  -> Signal dom LayerAddress           -- ^ extended address
  -> Signal dom (RowI8E ModelDimension)-- ^ row data (ModelDim mantissas + exp)
  -> Signal dom Bool                   -- ^ allDone (pulse when last segment row seen)
  -> Signal dom Bool                   -- ^ reset (layer change)
  -> Signal dom LayerWeightBuffer
layerWeightBufferController rowValid addr row allDone reset = bufS
  where
    stepEn = rowValid .||. reset

    -- State register
    bufS = regEn emptyLayerBuffer stepEn nextBuf

    -- Next-state function
    nextBuf =
      mux reset
        (pure emptyLayerBuffer)
        (apply <$> rowValid <*> addr <*> row <*> allDone <*> bufS)

    apply en a r done prev =
      let prev1 =
            if not en then prev else
            case seg a of
              SegQ      -> prev { qkv       = updateQKVOnce a r (qkv prev) }
              SegK      -> prev { qkv       = updateQKVOnce a r (qkv prev) }
              SegV      -> prev { qkv       = updateQKVOnce a r (qkv prev) }
              SegRmsAtt -> prev { rmsAttRow = r }
              SegRmsFfn -> prev { rmsFfnRow = r }
              SegW1     ->
                let i  = toIndexClamp @HiddenDimension (w1LoadedRows prev)
                    w1 = replace i r (w1Buf prev)
                in prev { w1Buf = w1, w1LoadedRows = w1LoadedRows prev + 1 }
              SegW3     ->
                let i  = toIndexClamp @HiddenDimension (w3LoadedRows prev)
                    w3 = replace i r (w3Buf prev)
                in prev { w3Buf = w3, w3LoadedRows = w3LoadedRows prev + 1 }
              -- Not handled here (kept as constants for now):
              SegWO     -> prev
              SegW2     -> prev
          -- Latch "all segments done" for this layer
          flg = fullyLoadedAll prev || done
      in prev1 { fullyLoadedAll = flg
               , qkv = (qkv prev1) { fullyLoaded = fullyLoaded (qkv prev1) || qkvDone a }
               }

    -- QKV done pulse detection: last row of last KV head in V
    qkvDone :: LayerAddress -> Bool
    qkvDone LayerAddress{..} =
      let kvLast :: Unsigned 8
          kvLast = fromInteger (natToNum @NumKeyValueHeads - 1)
      in seg == SegV && headIx == kvLast && rowIx == maxBound

-- Simple extractors (pure)
extractQKV :: LayerWeightBuffer -> QKVWeightBuffer
extractQKV = qkv

extractW1 :: LayerWeightBuffer -> MatI8E HiddenDimension ModelDimension
extractW1 = w1Buf

extractW3 :: LayerWeightBuffer -> MatI8E HiddenDimension ModelDimension
extractW3 = w3Buf

extractRmsAtt :: LayerWeightBuffer -> RowI8E ModelDimension
extractRmsAtt = rmsAttRow

extractRmsFfn :: LayerWeightBuffer -> RowI8E ModelDimension
extractRmsFfn = rmsFfnRow
