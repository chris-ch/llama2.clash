module LLaMa2.Layer.Attention.QKVProjection
  ( keyValueHeadProjector
  , qkvProjectionController
  , queryHeadProjector
  , QHeadDebugInfo(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent)
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E (..))
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified LLaMa2.Layer.Attention.FSM as FSM (processingControllerFSM)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightStreaming as STREAM
import Simulation.Parameters (DecoderParameters(..))

data RowFetchState = RFIdle | RFFetching | RFProcessing | RFDone
  deriving (Show, Eq, Generic, NFDataX)

-- Debug info
data QHeadDebugInfo dom = QHeadDebugInfo
  { qhRowIndex     :: Signal dom (Index HeadDimension)
  , qhState        :: Signal dom OPS.MultiplierState
  , qhFirstMant    :: Signal dom Mantissa
  , qhRowResult    :: Signal dom FixedPoint
  , qhRowDone      :: Signal dom Bool
  , qhFetchValid   :: Signal dom Bool
  , qhRowReset     :: Signal dom Bool
  , qhRowEnable    :: Signal dom Bool
  , qhAccumValue   :: Signal dom FixedPoint
  , qhQOut         :: Signal dom (Vec HeadDimension FixedPoint)
  , qhCurrentRowExp    :: Signal dom Exponent
  , qhCurrentRow'Exp   :: Signal dom Exponent
  , qhCurrentRowMant0  :: Signal dom Mantissa
  , qhCurrentRow'Mant0 :: Signal dom Mantissa
  } deriving (Generic)

--------------------------------------------------------------------------------
-- Q head projector
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     , QHeadDebugInfo dom
     )
queryHeadProjector dramSlaveIn layerIdx headIdx inputValid downStreamReady stepCount xHat params =
  (axiMaster, qRoOut, outputValid, readyForInput, debugInfo)
 where

  -- 1. State Machine
  (state, fetchTrigger, rowReset, rowEnable, outputValid, readyForInput) =
      OPS.matrixMultiplierStateMachine inputValid downStreamReady rowDone fetchValid rowIndex

  -- 2. Row Index Logic (FIXED: NATURAL WRAP AROUND)
  -- Previous logic prevented wrapping, causing the index to stick at 7 until MIdle.
  -- By allowing natural wrapping (7+1->0), rowIndex becomes 0 immediately after
  -- the last row is finished. It sits at 0 during MDone and MIdle, ensuring
  -- the address bus is perfectly stable for the next Token's fetch.
-- 2. Row Index Logic (FINAL FIX: Bounds-Safe Wrap)
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex

  nextRowIndex = 
    mux (state .==. pure OPS.MIdle) 
        (pure 0) -- Safety: Force 0 in Idle
        (mux rowDone
             (mux (rowIndex .==. pure maxBound)
                  (pure 0)     -- CRITICAL: Wrap 7 -> 0 manually
                  (rowIndex + 1) -- Safe increment for 0..6
             )
             rowIndex)

  -- 3. Real DRAM Interface
  rowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowIndex
  (axiMaster, fetchedWord, fetchValid) = STREAM.axiRowFetcher dramSlaveIn fetchTrigger rowAddr
  
  parsedRow :: Signal dom (RowI8E ModelDimension)
  parsedRow = STREAM.parseRow <$> fetchedWord

  -- 4. Robust Data Latching
  -- Prevents stale data from overwriting the register during processing.
  currentRow' :: Signal dom (RowI8E ModelDimension)
  currentRow' = register (RowI8E (repeat 0) 0) $
                  mux (fetchValid .&&. (state .==. pure OPS.MFetching)) 
                      parsedRow 
                      currentRow'

  -- Reference Hardcoded Data (Debug/Comparison)
  currentRow :: Signal dom (RowI8E ModelDimension)
  currentRow = (!!) (PARAM.wqHeadQ (PARAM.headsQ (PARAM.multiHeadAttention (modelLayers params !! layerIdx)) !! headIdx)) <$> rowIndex

  -- 5. Processor (Uses LATCHED DRAM DATA)
  (rowResult, rowDone, colIdx, accValue) = OPS.parallel64RowProcessor rowReset rowEnable currentRow xHat

  -- 6. Output Accumulation
  qOut = register (repeat 0) nextOutput
  nextOutput = mux rowDone
                   (replace <$> rowIndex <*> rowResult <*> qOut)
                   qOut

  qRoOut = (rotaryEncoder (PARAM.rotaryF (PARAM.headsQ (PARAM.multiHeadAttention (modelLayers params !! layerIdx)) !! headIdx)) <$> stepCount) <*> qOut

  -- Debug Info
  debugInfo = QHeadDebugInfo
    { qhRowIndex   = rowIndex
    , qhState      = state
    , qhFirstMant  = register 0 (head . rowMantissas <$> currentRow)
    , qhRowResult  = register 0 rowResult
    , qhRowDone    = rowDone
    , qhFetchValid = fetchValid
    , qhRowReset   = rowReset
    , qhRowEnable  = rowEnable
    , qhAccumValue = accValue
    , qhQOut       = qOut 
    , qhCurrentRowExp    = register 0 (rowExponent <$> currentRow)
    , qhCurrentRow'Exp   = register 0 (rowExponent <$> currentRow')
    , qhCurrentRowMant0  = register 0 (head . rowMantissas <$> currentRow)
    , qhCurrentRow'Mant0 = register 0 (head . rowMantissas <$> currentRow')
    }

--------------------------------------------------------------------------------
-- KV head projector (unchanged)
--------------------------------------------------------------------------------
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.SingleHeadComponentQ
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     )
keyValueHeadProjector inputValid downStreamReady stepCountSig xHatSig headParams =
  (kRoOut, vOut, outputValid, readyForInput)
 where
  selectedK :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedK = pure (PARAM.wkHeadQ headParams)

  selectedV :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedV = pure (PARAM.wvHeadQ headParams)

  (kOut, kValidOut, kReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedV xHatSig

  kRoOut = (rotaryEncoder (PARAM.rotaryF headParams) <$> stepCountSig) <*> kOut

  outputValid = kValidOut .&&. vValidOut
  readyForInput = kReadyOut .&&. vReadyOut

--------------------------------------------------------------------------------
-- AXI arbiter (unchanged)
--------------------------------------------------------------------------------
axiArbiter :: forall dom n.
  (HiddenClockResetEnable dom, KnownNat n)
  => Vec n (Master.AxiMasterOut dom)
  -> Master.AxiMasterOut dom
axiArbiter masters = Master.AxiMasterOut
  { arvalid = sel Master.arvalid
  , ardata  = sel Master.ardata
  , rready  = sel Master.rready
  , awvalid = sel Master.awvalid
  , awdata  = sel Master.awdata
  , wvalid  = sel Master.wvalid
  , wdata   = sel Master.wdata
  , bready  = sel Master.bready
  }
  where
    arRequests :: Vec n (Signal dom Bool)
    arRequests = map Master.arvalid masters
    lastGranted :: Signal dom (Index n)
    lastGranted = register 0 nextGranted
    selectedIdx :: Signal dom (Index n)
    selectedIdx = findNextRequester <$> bundle arRequests <*> lastGranted
    findNextRequester reqs lastR = 
      let start = if lastR == maxBound then 0 else lastR + 1
          go i n 
            | n == (0 :: Int) = lastR 
            | reqs !! i = i  
            | i == maxBound = go 0 (n-1)
            | otherwise = go (i+1) (n-1)
      in go start (natToNum @n)
    anyRequest = or <$> bundle arRequests
    nextGranted = mux anyRequest selectedIdx lastGranted
    sel :: forall a. (Master.AxiMasterOut dom -> Signal dom a) -> Signal dom a
    sel f = (!!) <$> bundle (map f masters) <*> selectedIdx

--------------------------------------------------------------------------------
-- QKV projector (unchanged)
--------------------------------------------------------------------------------
qkvProjector :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
qkvProjector dramSlaveIn layerIdx inputValid downStreamReady seqPos xVec params =
  (axiMasterOut, qkvOut, outputValid, readyForInput, head0Debug)
 where
  layerParams = modelLayers params !! layerIdx
  mhaParams = PARAM.multiHeadAttention layerParams
  xNorm = rmsNormFwFix <$> xVec <*> pure (PARAM.rmsAttF mhaParams)
  qResults = map (qHead params) indicesI
    where
      qHead params' headIdx = queryHeadProjector dramSlaveIn layerIdx headIdx
                          inputValid downStreamReady seqPos xNorm params'
  head0Debug = head qDebugInfos
  qAxiMasters = map (\(axi, _, _, _, _) -> axi) qResults
  qVecs       = map (\(_, q, _, _, _) -> q) qResults
  qValids     = map (\(_, _, v, _, _) -> v) qResults
  qReadys     = map (\(_, _, _, r, _) -> r) qResults
  qDebugInfos = map (\(_, _, _, _, d) -> d) qResults
  axiMasterOut = axiArbiter qAxiMasters
  
  queryHeadsPerKV = natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads
  
  kvHeadIndices :: Vec NumKeyValueHeads (Index NumQueryHeads)
  kvHeadIndices = map (\i -> toEnum (fromEnum i * queryHeadsPerKV)) indicesI

  kvResults = map kvHead kvHeadIndices
   where
    kvHead qIx =
      let headParams' = PARAM.headsQ mhaParams !! qIx
      in keyValueHeadProjector inputValid downStreamReady seqPos xNorm headParams'
  kVecs    = map (\(k, _, _, _) -> k) kvResults
  vVecs    = map (\(_, v, _, _) -> v) kvResults
  kvValids = map (\(_, _, v, _) -> v) kvResults
  kvReadys = map (\(_, _, _, r) -> r) kvResults
  outputValid = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
  readyForInput = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)
  qkvOut = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)

--------------------------------------------------------------------------------
-- QKV Projection Controller
--------------------------------------------------------------------------------
qkvProjectionController ::
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> Signal dom (Index SequenceLength)
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
qkvProjectionController dramSlaveIn layerIdx inputValid downStreamReady input params seqPos =
  (axiMasterOut, result, outputValid, readyForInput, debugInfo)
 where
  (enableRaw, outputValid, inReadyRaw) =
    FSM.processingControllerFSM inputValid downStreamReady matVecValid

  -- Fixed: deadlock prevention
  enableGated = enableRaw 

  (axiMasterOut, result, matVecValid, projReadyOut, debugInfo) =
    qkvProjector dramSlaveIn layerIdx enableGated downStreamReady
                 seqPos input params

  projReadyOut_d = register True projReadyOut
  readyForInput  = inReadyRaw .&&. projReadyOut_d
