-- File: LLaMa2/Layer/Attention/QKVProjection.hs (minimal fix)
module LLaMa2.Layer.Attention.QKVProjection
  ( keyValueHeadProjector
  , qkvProjectionController
  , queryHeadProjector
  , QHeadDebugInfo(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint, Mantissa)
import LLaMa2.Numeric.FixedPoint (rmsNormFwFix)
import LLaMa2.Numeric.Quantization (MatI8E, RowI8E)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified LLaMa2.Layer.Attention.FSM as FSM (processingControllerFSM)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightStreaming as STREAM

data RowFetchState = RFIdle | RFFetching | RFProcessing | RFDone
  deriving (Show, Eq, Generic, NFDataX)

-- Debug info for Q head weight streaming
data QHeadDebugInfo dom = QHeadDebugInfo
  { qhRowIndex     :: Signal dom (Index HeadDimension)
  , qhState        :: Signal dom OPS.MultiplierState
  , qhFirstMant    :: Signal dom Mantissa
  , qhRowResult    :: Signal dom FixedPoint
  , qhRowDone      :: Signal dom Bool
  , qhFetchValid   :: Signal dom Bool 
  , qhRowReset     :: Signal dom Bool  -- When is reset active?
  , qhRowEnable    :: Signal dom Bool  -- When is enable active?
  , qhAccumValue   :: Signal dom FixedPoint  -- What's in the accumulator?
  , qhQOut         :: Signal dom (Vec HeadDimension FixedPoint)  -- Current qOut register
 
  } deriving (Generic, NFDataX)

--------------------------------------------------------------------------------
-- Q head projector with DDR streaming (FIXED: latch fetched data)
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  =>
   Slave.AxiSlaveIn dom                     -- ^ DRAM interface
  -> Index NumLayers                          -- ^ layer index
  -> Unsigned NumQueryHeads                   -- ^ head index
  -> Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.SingleHeadComponentQ               -- ^ hardcoded (fallback)
  -> ( Master.AxiMasterOut dom                -- ^ To DRAM
     ,  Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     , QHeadDebugInfo dom
     )
queryHeadProjector dramSlaveIn layerIdx headIdx inputValid downStreamReady stepCountSig xHatSig headParams =
  (axiMaster, qRoOut, outputValid, readyForInput, debugInfo)
 where

  -- Row counter drives weight fetches
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex

  -- Calculate DDR address for current row
  rowAddr = STREAM.calculateRowAddress layerIdx STREAM.QMatrix headIdx <$> (fromIntegral . fromEnum <$> rowIndex)

  -- Fetch row when starting new row
  (axiMaster, fetchedWord, fetchValid) = STREAM.axiRowFetcher dramSlaveIn rowReset rowAddr

  -- Parse fetched word into row format
  currentRow :: Signal dom (RowI8E ModelDimension)
  currentRow = STREAM.parseRow <$> fetchedWord

  -- Row processor
  (rowResult, rowDone, colIdx, accValue) = OPS.parallel64RowProcessor rowReset rowEnable currentRow xHatSig

  -- State machine
  (state, rowReset, rowEnable, outputValid', readyForInput') =
    OPS.matrixMultiplierStateMachine inputValid downStreamReady rowDone rowIndex

  -- Row index sequencing
  nextRowIndex = mux (rowDone .&&. (rowIndex ./=. pure maxBound))
                     (rowIndex + 1)
                     (mux ((state .==. pure OPS.MDone) .&&. downStreamReady)
                          (pure 0)
                          rowIndex)

  -- Extract first mantissa for debugging
  firstMantissa = head . fst <$> currentRow

  -- Package debug info
  debugInfo = QHeadDebugInfo
    { qhRowIndex   = rowIndex
    , qhState      = state
    , qhFirstMant  = register 0 firstMantissa
    , qhRowResult  = register 0 rowResult
    , qhRowDone    = rowDone
    , qhFetchValid = fetchValid
    , qhRowReset   = rowReset
    , qhRowEnable  = rowEnable
    , qhAccumValue = accValue
    , qhQOut       = qOut
    }

  selectedQ :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedQ = pure (PARAM.wqHeadQ headParams) -- should be ramQ

  (qOut, outputValid, readyForInput) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedQ xHatSig

  qRoOut = (rotaryEncoder (PARAM.rotaryF headParams) <$> stepCountSig) <*> qOut

queryHeadProjector' :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom                     -- ^ DRAM interface
  -> Index NumLayers                          -- ^ layer index
  -> Unsigned NumQueryHeads                   -- ^ head index
  -> Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.SingleHeadComponentQ               -- ^ headParams (for rotary only)
  -> ( Master.AxiMasterOut dom                -- ^ To DRAM
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     , QHeadDebugInfo dom
     )
queryHeadProjector' dramSlaveIn layerIdx headIdx inputValid downStreamReady stepCountSig xHatSig headParams =
  (axiMaster, qRoOut, outputValid, readyForInput, debugInfo)
 where
  -- Row counter drives weight fetches
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex

  -- Calculate DDR address for current row
  rowAddr = STREAM.calculateRowAddress layerIdx STREAM.QMatrix headIdx <$> (fromIntegral . fromEnum <$> rowIndex)

  -- Fetch row when starting new row
  (axiMaster, fetchedWord, fetchValid) = STREAM.axiRowFetcher dramSlaveIn rowReset rowAddr

  -- Parse fetched word into row format
  currentRow :: Signal dom (RowI8E ModelDimension)
  currentRow = STREAM.parseRow <$> fetchedWord

  -- Row processor
  (rowResult, rowDone, colIdx, accValue) = OPS.parallel64RowProcessor rowReset rowEnable currentRow xHatSig

  -- State machine
  (state, rowReset, rowEnable, outputValid, readyForInput) =
    OPS.matrixMultiplierStateMachine inputValid downStreamReady rowDone rowIndex

  -- Row index sequencing
  nextRowIndex = mux (rowDone .&&. (rowIndex ./=. pure maxBound))
                     (rowIndex + 1)
                     (mux ((state .==. pure OPS.MDone) .&&. downStreamReady)
                          (pure 0)
                          rowIndex)

  -- Accumulate results
  qOut = register (repeat 0) nextOutput
  nextOutput = mux rowDone
                   (replace <$> rowIndex <*> rowResult <*> qOut)
                   qOut

  -- Apply rotary encoding (still using headParams)
  qRoOut = (rotaryEncoder (PARAM.rotaryF headParams) <$> stepCountSig) <*> qOut

  -- Extract first mantissa for debugging
  firstMantissa = head . fst <$> currentRow

  -- Package debug info
  debugInfo = QHeadDebugInfo
    { qhRowIndex   = rowIndex
    , qhState      = state
    , qhFirstMant  = register 0 firstMantissa
    , qhRowResult  = register 0 rowResult
    , qhRowDone    = rowDone
    , qhFetchValid = fetchValid
    , qhRowReset   = rowReset
    , qhRowEnable  = rowEnable
    , qhAccumValue = accValue
    , qhQOut       = qOut
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
  ( KnownNat n)
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

    arRequestsVecSig :: Signal dom (Vec n Bool)
    arRequestsVecSig = sequenceA arRequests

    selectedIdx :: Signal dom (Index n)
    selectedIdx = fmap
      (foldr (\(i,b) acc -> if b then i else acc) (0 :: Index n) . zip indicesI)
      arRequestsVecSig

    sel :: forall a. (Master.AxiMasterOut dom -> Signal dom a) -> Signal dom a
    sel field =
      let fieldVec :: Vec n (Signal dom a)
          fieldVec = map field masters
          fieldVecSig :: Signal dom (Vec n a)
          fieldVecSig = sequenceA fieldVec
      in (!!) <$> fieldVecSig <*> selectedIdx

--------------------------------------------------------------------------------
-- QKV projector and controller (unchanged)
--------------------------------------------------------------------------------
qkvProjector :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.MultiHeadAttentionComponentQ
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom  -- Debug info from head 0
     )
qkvProjector dramSlaveIn layerIdx inputValid downStreamReady seqPosSig xSig mhaParams =
  (axiMasterOut, qkvOut, outputValid, readyForInput, head0Debug)
 where
  xNorm = rmsNormFwFix <$> xSig <*> pure (PARAM.rmsAttF mhaParams)

  qResults = imap qHead (PARAM.headsQ mhaParams)
   where
    qHead :: Index NumQueryHeads
          -> PARAM.SingleHeadComponentQ
          -> ( Master.AxiMasterOut dom
             , Signal dom (Vec HeadDimension FixedPoint)
             , Signal dom Bool
             , Signal dom Bool 
             , QHeadDebugInfo dom
             )
    qHead headIdx = queryHeadProjector dramSlaveIn layerIdx (fromIntegral $ fromEnum headIdx)
                         inputValid downStreamReady seqPosSig xNorm

  -- Extract debug info from head 0
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
    kvHead :: Index NumQueryHeads
           -> ( Signal dom (Vec HeadDimension FixedPoint)
              , Signal dom (Vec HeadDimension FixedPoint)
              , Signal dom Bool
              , Signal dom Bool )
    kvHead qIx =
      let headParams' = PARAM.headsQ mhaParams !! qIx
      in keyValueHeadProjector inputValid downStreamReady seqPosSig xNorm headParams'

  kVecs    = map (\(k, _, _, _) -> k) kvResults
  vVecs    = map (\(_, v, _, _) -> v) kvResults
  kvValids = map (\(_, _, v, _) -> v) kvResults
  kvReadys = map (\(_, _, _, r) -> r) kvResults

  outputValid = (and <$> sequenceA qValids) .&&. (and <$> sequenceA kvValids)
  readyForInput = (and <$> sequenceA qReadys) .&&. (and <$> sequenceA kvReadys)

  qkvOut = bundle (sequenceA qVecs, sequenceA kVecs, sequenceA vVecs)

qkvProjectionController ::
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.MultiHeadAttentionComponentQ
  -> Signal dom (Index SequenceLength)
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool 
     , QHeadDebugInfo dom
     )
qkvProjectionController dramSlaveIn layerIdx inputValid downStreamReady input mhaParams seqPos =
  (axiMasterOut, result, outputValid, readyForInput, debugInfo)
 where
  (enableRaw, outputValid, inReadyRaw) =
    FSM.processingControllerFSM inputValid downStreamReady matVecValid

  (axiMasterOut, result, matVecValid, projReadyOut, debugInfo) =
    qkvProjector dramSlaveIn layerIdx enableGated downStreamReady
                 seqPos input mhaParams

  projReadyOut_d = register True projReadyOut
  enableGated    = enableRaw  .&&. projReadyOut_d
  readyForInput  = inReadyRaw .&&. projReadyOut_d
