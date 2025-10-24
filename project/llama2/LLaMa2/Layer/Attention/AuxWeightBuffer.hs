module LLaMa2.Layer.Attention.AuxWeightBuffer
  ( AuxWeightBuffer(..)
  , auxWeightBufferController
  , extractWO
  , extractW1, extractW2, extractW3
  , extractRmsFfn
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( ModelDimension, HeadDimension, HiddenDimension
  , NumQueryHeads )
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import LLaMa2.Memory.WeightLoaderAddressingExtended (LayerSeg(..), LayerAddress(..))

-- Counters as Index-typed to avoid undefined/out-of-range addressing
data WOBuffer = WOBuffer
  { woHeads  :: Vec NumQueryHeads (MatI8E ModelDimension HeadDimension)
  , woRowCnt :: Vec NumQueryHeads (Index ModelDimension)
  } deriving (Generic, NFDataX)

data FFNBuffer = FFNBuffer
  { w1    :: MatI8E HiddenDimension   ModelDimension
  , w2    :: MatI8E ModelDimension    HiddenDimension
  , w3    :: MatI8E HiddenDimension   ModelDimension
  , w1Cnt :: Index HiddenDimension
  , w2Cnt :: Index ModelDimension
  , w3Cnt :: Index HiddenDimension
  } deriving (Generic, NFDataX)

data AuxWeightBuffer = AuxWeightBuffer
  { wo   :: WOBuffer
  , ffn  :: FFNBuffer
  , rmsF :: RowI8E ModelDimension
  , fullyLoadedAll :: Bool
  } deriving (Generic, NFDataX)

-- Empty inits
emptyRowHD :: RowI8E HeadDimension;     emptyRowHD     = (repeat 0, 0)
emptyRowMD :: RowI8E ModelDimension;    emptyRowMD     = (repeat 0, 0)
emptyRowHID :: RowI8E HiddenDimension;  emptyRowHID    = (repeat 0, 0)

emptyWO :: WOBuffer
emptyWO = WOBuffer (repeat (repeat emptyRowHD)) (repeat 0)

emptyFFN :: FFNBuffer
emptyFFN = FFNBuffer (repeat emptyRowMD) (repeat emptyRowHID) (repeat emptyRowMD) 0 0 0

emptyAux :: AuxWeightBuffer
emptyAux = AuxWeightBuffer emptyWO emptyFFN emptyRowMD False

-- Safe increment that saturates at maxBound (shouldn't be hit if rowsInSeg is correct)
incIdx :: (Eq (Index n), Bounded (Index n), Enum (Index n)) => Index n -> Index n
incIdx i = if i == maxBound then maxBound else succ i

-- Controller
auxWeightBufferController ::
  HiddenClockResetEnable dom =>
  Signal dom Bool                      ->  -- mdRowValid
  Signal dom Bool                      ->  -- hdRowValid (WO)
  Signal dom Bool                      ->  -- hidRowValid (W2)
  Signal dom LayerAddress              ->  -- addr (seg/headIx)
  Signal dom (RowI8E ModelDimension)   ->  -- mdRow
  Signal dom (RowI8E HeadDimension)    ->  -- hdRow
  Signal dom (RowI8E HiddenDimension)  ->  -- hidRow
  Signal dom Bool                      ->  -- allSegmentsDone pulse
  Signal dom Bool                      ->  -- reset
  Signal dom AuxWeightBuffer
auxWeightBufferController mdV hdV hidV addr mdRow hdRow hidRow allDone reset = bufS
 where
  stepEn = mdV .||. hdV .||. hidV .||. reset
  bufS   = regEn emptyAux stepEn nextBuf

  nextBuf =
    mux reset (pure emptyAux)
      (apply <$> mdV <*> hdV <*> hidV <*> addr <*> mdRow <*> hdRow <*> hidRow <*> allDone <*> bufS)

  apply mdv hdv hidv LayerAddress{..} mdR hdR hidR done prev =
    let prev1 = case seg of
          SegWO | hdv ->
            case toIndexMaybe @NumQueryHeads headIx of
              Nothing -> prev
              Just hi ->
                let rIx  = woRowCnt (wo prev) !! hi                       -- Index ModelDimension
                    matH = woHeads (wo prev)  !! hi
                    matH' = replace rIx hdR matH                             -- safe replace
                    cntV  = replace hi (incIdx rIx) (woRowCnt (wo prev))
                in prev { wo = (wo prev) { woHeads = replace hi matH' (woHeads (wo prev))
                                          , woRowCnt = cntV } }
          SegW1 | mdv ->
            let i   = w1Cnt (ffn prev)                                      -- Index HiddenDimension
                w1' = replace i mdR (w1 (ffn prev))
            in prev { ffn = (ffn prev){ w1 = w1', w1Cnt = incIdx i } }
          SegW2 | hidv ->
            let i   = w2Cnt (ffn prev)                                      -- Index ModelDimension
                w2' = replace i hidR (w2 (ffn prev))
            in prev { ffn = (ffn prev){ w2 = w2', w2Cnt = incIdx i } }
          SegW3 | mdv ->
            let i   = w3Cnt (ffn prev)                                      -- Index HiddenDimension
                w3' = replace i mdR (w3 (ffn prev))
            in prev { ffn = (ffn prev){ w3 = w3', w3Cnt = incIdx i } }
          SegRmsFfn | mdv ->
            prev { rmsF = mdR }
          _ -> prev
        fl = fullyLoadedAll prev1 || done
    in prev1 { fullyLoadedAll = fl }

toIndexMaybe :: forall m . KnownNat m => Unsigned 8 -> Maybe (Index m)
toIndexMaybe u =
  let lim = natToNum @m
  in if toInteger u < lim then Just (toEnum (fromIntegral u)) else Nothing

-- Extractors
extractWO :: AuxWeightBuffer -> Index NumQueryHeads -> MatI8E ModelDimension HeadDimension
extractWO b hi = woHeads (wo b) !! hi
extractW1 :: AuxWeightBuffer -> MatI8E HiddenDimension   ModelDimension; extractW1 b = w1 (ffn b)
extractW2 :: AuxWeightBuffer -> MatI8E ModelDimension    HiddenDimension; extractW2 b = w2 (ffn b)
extractW3 :: AuxWeightBuffer -> MatI8E HiddenDimension   ModelDimension; extractW3 b = w3 (ffn b)
extractRmsFfn :: AuxWeightBuffer -> RowI8E ModelDimension; extractRmsFfn = rmsF
