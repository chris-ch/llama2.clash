module Simulation.DRAMBackedAxiSlave
  ( DRAMConfig(..)
  , WordData
  , createDRAMBackedAxiSlaveFromVec
  , createDRAMBackedAxiSlave
  , buildMemoryFromParams
  ) where

import Clash.Prelude
import qualified Prelude as P

import LLaMa2.Memory.AXI.Types
    ( AxiW(wdata),
      AxiAW(AxiAW, awid, awaddr, awlen),
      AxiR(..),
      AxiAR(AxiAR, arid, arlen, araddr),
      AxiB(AxiB) )
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import LLaMa2.Types.ModelConfig
    ( NumKeyValueHeads, NumQueryHeads, NumLayers )
import qualified Simulation.Parameters as PARAM
import Clash.Sized.Vector (unsafeFromList)
import qualified LLaMa2.Memory.WeightsLayout as Layout
import Clash.Debug (trace)

-- | Timing configuration
-- First field is EXTRA read latency (beyond the inherent 1-cycle RAM).
data DRAMConfig = DRAMConfig
  { rValidDelay  :: Int  -- extra cycles from AR accept to first RVALID
  , arReadyDelay :: Int  -- cycles before accepting next AR
  , rBeatDelay   :: Int  -- extra cycles between R beats
  } deriving (Show, Eq)

type WordData = BitVector 512  -- 64 bytes per beat

-- ============================================================================
-- Top-level constructors
-- ============================================================================

-- | Convenience constructor keeping the historical default 64Ki-word depth.
-- Use 'createDRAMBackedAxiSlaveFromVec' for any other depth.
createDRAMBackedAxiSlave ::
  forall dom.
  HiddenClockResetEnable dom =>
  PARAM.DecoderParameters ->
  Master.AxiMasterOut dom ->
  Slave.AxiSlaveIn dom
createDRAMBackedAxiSlave params =
  createDRAMBackedAxiSlaveFromVec (DRAMConfig 1 0 1) (buildMemoryFromParams @65536 params)

-- ============================================================================
-- Build a DRAM image (generic depth)
-- ============================================================================

-- | Build a DRAM image from model parameters, trimmed/padded to n words.
--   Section order MUST match rowAddressCalculator in WeightsLayout.
buildMemoryFromParams :: forall n. KnownNat n => PARAM.DecoderParameters -> Vec n WordData
buildMemoryFromParams params =
  let nI = natToNum @n :: Int
  in unsafeFromList $ P.take nI $ allWords P.++ P.repeat 0
  where
    allWords = embeddingWords
            P.++ rmsFinalWords
            P.++ rotaryWords
            P.++ P.concatMap layerWords [0..numLayers-1]

    numLayers  = natToNum @NumLayers :: Int
    numQHeads  = natToNum @NumQueryHeads :: Int
    numKVHeads = natToNum @NumKeyValueHeads :: Int

    embedding       = PARAM.modelEmbedding params
    embeddingWords  = Layout.matrixMultiWordPacker (PARAM.vocabularyQ embedding)
    rmsFinalWords   = Layout.fixedPointVecPacker (PARAM.rmsFinalWeightF embedding)

    rotaryWords =
      let rotary   = PARAM.rotaryEncoding params  -- Get from top level
          cosWords = P.concatMap Layout.fixedPointVecPacker (toList (PARAM.freqCosF rotary))
          sinWords = P.concatMap Layout.fixedPointVecPacker (toList (PARAM.freqSinF rotary))
          totalB   = (P.length cosWords + P.length sinWords) * 64
          padW     = (Layout.align64 totalB - totalB) `div` 64
      in cosWords P.++ sinWords P.++ P.replicate padW 0

    layerWords li =
      let layer = PARAM.modelLayers params !! li
          mha   = PARAM.multiHeadAttention layer
          ffn   = PARAM.feedforwardNetwork layer
          
          -- Q matrices: all 8 Q heads
          qWords = P.concatMap (\h -> Layout.matrixMultiWordPacker 
                    (PARAM.qMatrix (PARAM.qHeads mha !! h)))
                  [0..numQHeads-1]
          
          -- K matrices: directly from 4 KV heads (NO MAPPING!)
          kWords = P.concatMap (\kvh -> Layout.matrixMultiWordPacker 
                    (PARAM.kMatrix (PARAM.kvHeads mha !! kvh)))
                  [0..numKVHeads-1]
          
          -- V matrices: directly from 4 KV heads (NO MAPPING!)
          vWords = P.concatMap (\kvh -> Layout.matrixMultiWordPacker 
                    (PARAM.vMatrix (PARAM.kvHeads mha !! kvh)))
                  [0..numKVHeads-1]
          
          -- WO still per Q head
          woWords = P.concatMap (\h -> Layout.matrixMultiWordPacker (PARAM.mWoQ mha !! h))
                    [0..numQHeads-1]
      in
        Layout.fixedPointVecPacker
          (PARAM.rmsAttF mha)
          P.++ qWords P.++ kWords P.++ vWords P.++ woWords
          P.++ Layout.fixedPointVecPacker (PARAM.fRMSFfnF ffn)
          P.++ Layout.matrixMultiWordPacker (PARAM.fW1Q ffn)
          P.++ Layout.matrixMultiWordPacker (PARAM.fW2Q ffn)
          P.++ Layout.matrixMultiWordPacker (PARAM.fW3Q ffn)

-- ============================================================================
-- AXI Slave (depth-generic)
-- ============================================================================

data ReadState  = RIdle | RProcessing (Index 256) (Index 256)
  deriving (Generic, NFDataX, Show, Eq)

data WriteState = WIdle | WProcessing (Index 256) (Index 256)
  deriving (Generic, NFDataX, Show, Eq)

-- | Generic-depth DRAM-backed AXI slave (read+write for tests; writes optional in HW).
createDRAMBackedAxiSlaveFromVec :: forall dom n.
  (HiddenClockResetEnable dom, KnownNat n)
  => DRAMConfig
  -> Vec n WordData
  -> Master.AxiMasterOut dom
  -> Slave.AxiSlaveIn dom
createDRAMBackedAxiSlaveFromVec config initVec masterIn =
  Slave.AxiSlaveIn
    { arready = arreadySig
    , rvalid  = rvalidSig
    , rdata   = rdataSigTraced
    , awready = awreadySig
    , wready  = wreadySig
    , bvalid  = bvalidSig
    , bdata   = bdataSig
    }
  where
    -- ==================== Shared RAM (generic depth) ====================
    ramReadAddr :: Signal dom (Index n)
    ramReadAddr = readAddrIdx

    writeData :: Signal dom WordData
    writeData = wdata <$> Master.wdata masterIn

    writeM :: Signal dom (Maybe (Index n, WordData))
    writeM = mux wHandshake (Just <$> bundle (writeAddrIdx, writeData))
                            (pure Nothing)

    ramOut :: Signal dom WordData
    ramOut = blockRam initVec ramReadAddr writeM

    readAddrEn  :: Signal dom Bool
    readAddrEn  = (\case RProcessing{} -> True; _ -> False) <$> readState
    ramOutValid :: Signal dom Bool
    ramOutValid = register False readAddrEn

    nWordsU32 :: Unsigned 32
    nWordsU32 = fromIntegral (natToNum @n :: Int)

    readState :: Signal dom ReadState
    readState = register RIdle nextReadState

    capturedAR :: Signal dom AxiAR
    capturedAR = regEn (AxiAR 0 0 0 0 0) arAccepted (Master.ardata masterIn)

    arDelayCounter :: Signal dom (Index 16)
    arDelayCounter = register 0 nextARDelay

    rDelayCounter :: Signal dom (Index 16)
    rDelayCounter = register 0 nextRDelay

    arreadySig :: Signal dom Bool
    arreadySig = (readState .==. pure RIdle) .&&. (arDelayCounter .==. pure 0)

    arAccepted :: Signal dom Bool
    arAccepted = Master.arvalid masterIn .&&. arreadySig

    currentBeat :: Signal dom (Index 256)
    currentBeat = (\case RProcessing b _ -> b; _ -> 0) <$> readState

    rHandshake :: Signal dom Bool
    rHandshake = rvalidSig .&&. Master.rready masterIn

    nextReadState :: Signal dom ReadState
    nextReadState =
      mux arAccepted
          ((\ar -> RProcessing 0 (fromInteger $ toInteger $ arlen ar :: Index 256))
            <$> Master.ardata masterIn)
      $ mux (isProc <$> readState .&&. rHandshake)
          (advance <$> readState)
          readState
      where
        isProc (RProcessing _ _) = True
        isProc _                 = False
        advance (RProcessing b l) | b >= l    = RIdle
                                  | otherwise = RProcessing (b+1) l
        advance s = s

    beatOffsetBytes :: Signal dom (Unsigned 32)
    beatOffsetBytes = (`shiftL` 6) . fromIntegral <$> currentBeat

    currentAddress :: Signal dom (Unsigned 32)
    currentAddress = (+) <$> (araddr <$> capturedAR) <*> beatOffsetBytes

    addrWordU :: Signal dom (Unsigned 32)
    addrWordU = (`shiftR` 6) <$> currentAddress

    readAddrIdx :: Signal dom (Index n)
    readAddrIdx =
      (fromIntegral :: Unsigned 32 -> Index n) <$> ((`mod` nWordsU32) <$> addrWordU)

    rvalidSig :: Signal dom Bool
    rvalidSig = (\s del ok -> case s of
                    RProcessing{} -> (del == 0) && ok
                    _             -> False)
                <$> readState <*> rDelayCounter <*> ramOutValid

    rdataSig :: Signal dom AxiR
    rdataSig = (\s ar dat ->
                  let isLast = case s of
                                 RProcessing b l -> b >= l
                                 _               -> False
                  in AxiR dat 0 isLast (arid ar)
               ) <$> readState <*> capturedAR <*> ramOut

    nextARDelay :: Signal dom (Index 16)
    nextARDelay =
      mux arAccepted
          (pure (fromIntegral (arReadyDelay config)))
      $ mux (arDelayCounter .>. pure 0)
          (arDelayCounter - 1)
          arDelayCounter

    -- Load exactly rValidDelay/rBeatDelay (BRAM already gives +1)
    nextRDelay :: Signal dom (Index 16)
    nextRDelay =
      let loadAfterAR   = fromIntegral (rValidDelay config)
          loadAfterBeat = fromIntegral (rBeatDelay  config)
          processing    = (\case RProcessing{} -> True; _ -> False) <$> readState
      in mux arAccepted
            (pure loadAfterAR)
       $ mux (rHandshake .&&. processing)
            (pure loadAfterBeat)
       $ mux (rDelayCounter .>. pure 0)
            (rDelayCounter - 1)
            rDelayCounter

    writeState :: Signal dom WriteState
    writeState = register WIdle nextWriteState

    capturedAW :: Signal dom AxiAW
    capturedAW = regEn (AxiAW 0 0 0 0 0) awAccepted (Master.awdata masterIn)

    awreadySig :: Signal dom Bool
    awreadySig = writeState .==. pure WIdle

    awAccepted :: Signal dom Bool
    awAccepted = Master.awvalid masterIn .&&. awreadySig

    -- Accept W either when already processing or in SAME cycle as AW is accepted
    wreadySig :: Signal dom Bool
    wreadySig = ((\case WProcessing{} -> True; _ -> False) <$> writeState) .||. awAccepted

    wBeat :: Signal dom (Index 256)
    wBeat = (\case WProcessing b _ -> b; _ -> 0) <$> writeState

    writeBaseAddr :: Signal dom (Unsigned 32)
    writeBaseAddr = mux awAccepted (awaddr <$> Master.awdata masterIn)
                                 (awaddr <$> capturedAW)

    writeBeatIdx :: Signal dom (Index 256)
    writeBeatIdx = mux awAccepted (pure 0) wBeat

    writeAddress :: Signal dom (Unsigned 32)
    writeAddress = (+) <$> writeBaseAddr <*> ((`shiftL` 6) . fromIntegral <$> writeBeatIdx)

    writeAddrIdx :: Signal dom (Index n)
    writeAddrIdx =
      (fromIntegral :: Unsigned 32 -> Index n)
      <$> ((`mod` nWordsU32) <$> ( (`shiftR` 6) <$> writeAddress ))

    wHandshake :: Signal dom Bool
    wHandshake = wreadySig .&&. Master.wvalid masterIn

    firstBeatAccepted :: Signal dom Bool
    firstBeatAccepted = awAccepted .&&. Master.wvalid masterIn

    -- QUALIFY last-beat-by-length with WProcessing to avoid early BVALID
    wProcessing :: Signal dom Bool
    wProcessing = (\case WProcessing{} -> True; _ -> False) <$> writeState

    isLastWBeat :: Signal dom Bool
    isLastWBeat = (\case
        WProcessing b l -> b >= l
        _ -> False) <$> writeState

    nextWriteState :: Signal dom WriteState
    nextWriteState =
      mux awAccepted
          ((\aw fb ->
              let l :: Index 256
                  l = fromInteger $ toInteger $ awlen aw
                  b0 = if fb then 1 else 0
              in WProcessing b0 l) <$> Master.awdata masterIn <*> firstBeatAccepted)
      $ mux (isProc <$> writeState .&&. wHandshake)
          (advance <$> writeState)
          writeState
      where
        isProc (WProcessing _ _) = True
        isProc _                 = False
        advance (WProcessing b l) | b >= l    = WIdle
                                  | otherwise = WProcessing (b+1) l
        advance s = s

    bvalidReg :: Signal dom Bool
    bvalidReg = register False nextBValid

    bvalidSig :: Signal dom Bool
    bvalidSig = bvalidReg

    -- AW+W same-cycle last-beat only for single-beat writes (len==0)
    singleBeatNow :: Signal dom Bool
    singleBeatNow = firstBeatAccepted .&&. ((\aw -> awlen aw == 0) <$> Master.awdata masterIn)

    lastBeatAccepted :: Signal dom Bool
    lastBeatAccepted =
      -- Only allow the length-based path while in WProcessing
      (wHandshake .&&. wProcessing .&&. isLastWBeat) .||. singleBeatNow

    nextBValid :: Signal dom Bool
    nextBValid =
      mux lastBeatAccepted
          (pure True)
      $ mux (bvalidReg .&&. Master.bready masterIn)
          (pure False)
          bvalidReg

    bdataSig :: Signal dom AxiB
    bdataSig = AxiB 0 . awid <$> capturedAW

    traceRead :: Signal dom AxiR -> Signal dom AxiR
    traceRead rdataIn = 
      go <$> rvalidSig <*> currentAddress <*> readAddrIdx <*> rdataIn
        where
          go True addr idx r = 
            trace ("[DRAMBackedAxisSlave] addr=" P.++ show addr 
                    P.++ " wordIdx=" P.++ show idx
                    P.++ " data[0..3]=" P.++ show (P.take 4 $ toList $ unpack (rdata r) :: [BitVector 8]))
            r
          go False _ _ r = r

    rdataSigTraced :: Signal dom AxiR
    rdataSigTraced = traceRead rdataSig
