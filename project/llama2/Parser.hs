module Parser (parseLLaMa2ConfigFile) where

import Prelude

import qualified Data.Binary.Get as BG
import qualified Data.ByteString as BS
import qualified Foreign as F
import qualified Data.Vector.Unboxed as V
import qualified Clash.Sized.Vector as CV
import qualified Clash.Prelude as C

import GHC.IO (unsafePerformIO)
import Control.Monad (replicateM_)
import LLaMa2.Types.LayerData
    (
      SingleHeadComponent(SingleHeadComponent, rotary, wqHead, wkHead,
                          wvHead),
      RotaryEncodingComponent(RotaryEncodingComponent, freqSin, freqCos),
      EmbeddingComponent(EmbeddingComponent, rmsFinalWeight, vocabulary),
      CArray2D(CArray2D), MultiHeadAttentionComponent (..), FeedForwardNetworkComponent (..) )

import LLaMa2.Types.ModelConfig 
    (
      HeadDimension,
      SequenceLength,
      VocabularySize,
      NumQueryHeads,
      NumLayers,
      NumKeyValueHeads,
      HiddenDimension,
      ModelDimension,
      RotaryPositionalEmbeddingDimension
      )
import Simulation.Parameters (DecoderParameters (..), TransformerLayerComponent (..), MultiHeadAttentionComponentQ, FeedForwardNetworkComponentQ, quantizeMHA, quantizeFFN, quantizeEmbedding)


-- ============================================================================
-- File Parsing
-- ============================================================================

readVector :: Int -> BG.Get (V.Vector Float)
readVector count = do
  byteData <- BG.getByteString (count * 4)
  return $! unsafePerformIO $ do
    BS.useAsCString byteData $ \ptr -> do
      let floatPtr = F.castPtr ptr :: F.Ptr Float
      V.generateM count (F.peekElemOff floatPtr)

readVec1D :: forall n. C.KnownNat n => BG.Get (C.Vec n Float)
readVec1D = do
    let total = C.snatToNum (C.SNat :: C.SNat n)
    vec <- readVector total
    return $ CV.unsafeFromList (V.toList vec)

readVec2D :: forall n m. (C.KnownNat n, C.KnownNat m) => BG.Get (C.Vec n (C.Vec m Float))
readVec2D = do
    let n = C.snatToNum (C.SNat :: C.SNat n)
        m = C.snatToNum (C.SNat :: C.SNat m)
        total = n * m
    vec <- readVector total
    let floatList = V.toList vec
        chunks = chunksOf m floatList
        vecs = map CV.unsafeFromList chunks
    return $ CV.unsafeFromList vecs
  where
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

readVec3D :: forall n m p. (C.KnownNat n, C.KnownNat m, C.KnownNat p) => BG.Get (C.Vec n (C.Vec m (C.Vec p Float)))
readVec3D = do
    let n = C.snatToNum (C.SNat :: C.SNat n)
        m = C.snatToNum (C.SNat :: C.SNat m)
        p = C.snatToNum (C.SNat :: C.SNat p)
        total = n * m * p
    vec <- readVector total
    let floatList = V.toList vec
        innerChunks = chunksOf p floatList
        innerVecs = map CV.unsafeFromList innerChunks
        middleChunks = chunksOf m innerVecs
        middleVecs = map CV.unsafeFromList middleChunks
    return $ CV.unsafeFromList middleVecs
  where
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

readVec4D :: forall m n p q. (C.KnownNat m, C.KnownNat n, C.KnownNat p, C.KnownNat q) => BG.Get (C.Vec m (C.Vec n (C.Vec p (C.Vec q Float))))
readVec4D = do
    let m = C.snatToNum (C.SNat :: C.SNat m)
        n = C.snatToNum (C.SNat :: C.SNat n)
        p = C.snatToNum (C.SNat :: C.SNat p)
        q = C.snatToNum (C.SNat :: C.SNat q)
        total = m * n * p * q
    vec <- readVector total
    let floatList = V.toList vec
        -- First chunk into q-sized vectors
        innerChunks = chunksOf q floatList
        innerVecs = map CV.unsafeFromList innerChunks
        -- Then chunk into p groups of q-sized vectors
        middleChunks = chunksOf p innerVecs
        middleVecs = map CV.unsafeFromList middleChunks
        -- Then chunk into n groups of p×q-sized tensors
        outerChunks = chunksOf n middleVecs
        outerVecs = map CV.unsafeFromList outerChunks
        -- Finally chunk into m groups
    return $ CV.unsafeFromList outerVecs
  where
    chunksOf _ [] = []
    chunksOf k xs = take k xs : chunksOf k (drop k xs)

parseLLaMa2ConfigFile :: BG.Get DecoderParameters
parseLLaMa2ConfigFile = do
  replicateM_ 7 BG.getInt32le
  tokenEmbeddingTable' <- readVec2D @VocabularySize @ModelDimension
  rmsAttWeight'        <- readVec2D @NumLayers @ModelDimension
  wq'                  <- readVec4D @NumLayers @NumQueryHeads     @HeadDimension  @ModelDimension
  wk'                  <- readVec4D @NumLayers @NumKeyValueHeads  @HeadDimension  @ModelDimension
  wv'                  <- readVec4D @NumLayers @NumKeyValueHeads  @HeadDimension  @ModelDimension
  wo'                  <- readVec3D @NumLayers @ModelDimension    @ModelDimension
  rmsFfnWeight'        <- readVec2D @NumLayers @ModelDimension
  w1'                  <- readVec3D @NumLayers @HiddenDimension   @ModelDimension
  w2'                  <- readVec3D @NumLayers @ModelDimension    @HiddenDimension
  w3'                  <- readVec3D @NumLayers @HiddenDimension   @ModelDimension
  rmsFinalWeight'      <- readVec1D @ModelDimension
  freqCisReal'         <- readVec2D @SequenceLength @RotaryPositionalEmbeddingDimension
  freqCisImag'         <- readVec2D @SequenceLength @RotaryPositionalEmbeddingDimension

  let
    -- Build Float embedding, then quantize
    embeddingFloat = EmbeddingComponent
      { vocabulary     = CArray2D tokenEmbeddingTable'
      , rmsFinalWeight = rmsFinalWeight'
      }
    embeddingQ = quantizeEmbedding embeddingFloat

    layer :: C.Index NumLayers -> TransformerLayerComponent
    layer lIdx =
      let
        sha :: C.Index NumQueryHeads -> SingleHeadComponent
        sha hIdx =
          let nQ  = C.snatToNum (C.SNat @NumQueryHeads)     :: Int
              nKV = C.snatToNum (C.SNat @NumKeyValueHeads)  :: Int
              kvMul = max 1 (nQ `div` nKV)
              kvIdxInt = fromIntegral hIdx `div` kvMul
              kvIdx :: C.Index NumKeyValueHeads
              kvIdx = fromInteger (toInteger kvIdxInt)
          in SingleHeadComponent
               { wqHead = CArray2D $ (wq' C.!! lIdx) C.!! hIdx
               , wkHead = CArray2D $ (wk' C.!! lIdx) C.!! kvIdx
               , wvHead = CArray2D $ (wv' C.!! lIdx) C.!! kvIdx
               , rotary  = RotaryEncodingComponent
                   { freqCos = CArray2D freqCisReal'
                   , freqSin = CArray2D freqCisImag'
                   }
               }
        woLayer :: C.Vec ModelDimension (C.Vec ModelDimension Float)
        woLayer = wo' C.!! lIdx
        headBlock :: C.Index NumQueryHeads -> CArray2D ModelDimension HeadDimension
        headBlock hIdx =
          let base :: Int
              base = fromIntegral hIdx * C.snatToNum (C.SNat @HeadDimension)
              rowSlice :: C.Vec ModelDimension Float -> C.Vec HeadDimension Float
              rowSlice row =
                C.map
                  (\off -> row C.!! (toEnum (base + fromIntegral off) :: C.Index ModelDimension))
                  (C.indicesI @HeadDimension)
          in CArray2D (C.map rowSlice woLayer)

        mWoVec :: C.Vec NumQueryHeads (CArray2D ModelDimension HeadDimension)
        mWoVec = C.map headBlock (C.indicesI @NumQueryHeads)

        mhaFloat :: MultiHeadAttentionComponent
        mhaFloat = MultiHeadAttentionComponent
          { heads  = C.map sha (C.indicesI :: C.Vec NumQueryHeads (C.Index NumQueryHeads))
          , mWo    = mWoVec
          , rmsAtt = rmsAttWeight' C.!! lIdx
          }

        mhaQ :: MultiHeadAttentionComponentQ
        mhaQ = quantizeMHA mhaFloat

        ffnFloat :: FeedForwardNetworkComponent
        ffnFloat = FeedForwardNetworkComponent
          { fW1     = CArray2D $ w1' C.!! lIdx
          , fW2     = CArray2D $ w2' C.!! lIdx
          , fW3     = CArray2D $ w3' C.!! lIdx
          , fRMSFfn = rmsFfnWeight' C.!! lIdx
          }
        ffnQ :: FeedForwardNetworkComponentQ
        ffnQ = quantizeFFN ffnFloat

      in TransformerLayerComponent
           { multiHeadAttention = mhaQ
           , feedforwardNetwork = ffnQ
           }

    decoder = DecoderParameters
      { modelEmbedding = embeddingQ
      , modelLayers    = C.map layer (C.indicesI :: C.Vec NumLayers (C.Index NumLayers))
      }

  return decoder
