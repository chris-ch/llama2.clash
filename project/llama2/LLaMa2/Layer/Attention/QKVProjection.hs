-- File: LLaMa2/Layer/Attention/QKVProjection.hs (minimal fix)
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
    , qhCurrentRowExp    :: Signal dom Exponent  -- Exponent from hardcoded
    , qhCurrentRow'Exp   :: Signal dom Exponent  -- Exponent from DRAM
    , qhCurrentRowMant0  :: Signal dom Mantissa  -- First mantissa hardcoded
    , qhCurrentRow'Mant0 :: Signal dom Mantissa  -- First mantissa DRAM
  } deriving (Generic)

--------------------------------------------------------------------------------
-- Q head projector with DDR streaming (FIXED: latch fetched data)
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom                     -- ^ DRAM interface
  -> Index NumLayers                          -- ^ layer index
  -> Index NumQueryHeads                   -- ^ head index
  -> Signal dom Bool                          -- ^ inputValid
  -> Signal dom Bool                          -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters               -- ^ hardcoded (fallback)
  -> ( Master.AxiMasterOut dom                -- ^ To DRAM
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool                        -- ^ outputValid
     , Signal dom Bool                        -- ^ readyForInput
     , QHeadDebugInfo dom
     )
queryHeadProjector dramSlaveIn layerIdx headIdx inputValid downStreamReady stepCount xHat params =
  (axiMaster, qRoOut, outputValid, readyForInput, debugInfo)
 where

  -- Row counter drives weight fetches
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex

  -- Calculate DDR address for current row
  rowAddr = STREAM.calculateRowAddress STREAM.QMatrix layerIdx headIdx <$> rowIndex

  -- Fetch row when starting new row
  fetchedWord :: Signal dom (BitVector 512)
  (axiMaster, fetchedWord, fetchValid) = STREAM.axiRowFetcher dramSlaveIn fetchTrigger rowAddr
 
  -- Parse the LATCHED word
  currentRow' :: Signal dom (RowI8E ModelDimension)
  currentRow' = STREAM.parseRow <$> fetchedWord

  -- Fetch current row from the runtime matrix
  currentRow :: Signal dom (RowI8E ModelDimension)
  currentRow = (!!) (PARAM.wqHeadQ (PARAM.headsQ (PARAM.multiHeadAttention (modelLayers params !! layerIdx)) !! headIdx)) <$> rowIndex

  -- Row processor
  (rowResult, rowDone, colIdx, accValue) = OPS.parallel64RowProcessor rowReset rowEnable currentRow xHat

  -- State machine
  (state, fetchTrigger, rowReset, rowEnable, outputValid, readyForInput) =
      OPS.matrixMultiplierStateMachine inputValid downStreamReady rowDone (pure True) rowIndex

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

  qRoOut = (rotaryEncoder (PARAM.rotaryF (PARAM.headsQ (PARAM.multiHeadAttention (modelLayers params !! layerIdx)) !! headIdx)) <$> stepCount) <*> qOut

  -- Extract first mantissa for debugging
  firstMantissa = head . rowMantissas <$> currentRow

  currentRowExp = rowExponent <$> currentRow
  currentRow'Exp = rowExponent <$> currentRow'
  currentRowMant0 = head . rowMantissas <$> currentRow
  currentRow'Mant0 = head . rowMantissas <$> currentRow'

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
    , qhCurrentRowExp    = register 0 currentRowExp
    , qhCurrentRow'Exp   = register 0 currentRow'Exp
    , qhCurrentRowMant0  = register 0 currentRowMant0
    , qhCurrentRow'Mant0 = register 0 currentRow'Mant0
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
-- AXI arbiter
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

    -- Priority with fairness: increment when we grant
    lastGranted :: Signal dom (Index n)
    lastGranted = register 0 nextGranted
    
    -- Find next requester starting from lastGranted+1
    -- This ensures fairness while being responsive
    selectedIdx :: Signal dom (Index n)
    selectedIdx = findNextRequester <$> bundle arRequests <*> lastGranted
    
    findNextRequester :: Vec n Bool -> Index n -> Index n
    findNextRequester reqs lastR = 
      let start = if lastR == maxBound then 0 else lastR + 1
          -- Check each position starting from 'start'
          go i n 
            | n == (0 :: Int) = lastR  -- No requesters, keep last
            | reqs !! i = i   -- Found one!
            | i == maxBound = go 0 (n-1)
            | otherwise = go (i+1) (n-1)
      in go start (natToNum @n)
    
    -- Update lastGranted only when we actually grant
    anyRequest = or <$> bundle arRequests
    nextGranted = mux anyRequest selectedIdx lastGranted

    sel :: forall a. (Master.AxiMasterOut dom -> Signal dom a) -> Signal dom a
    sel f = (!!) <$> bundle (map f masters) <*> selectedIdx

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
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom ( Vec NumQueryHeads    (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)
                  , Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom  -- Debug info from head 0
     )
qkvProjector dramSlaveIn layerIdx inputValid downStreamReady seqPos xVec params =
  (axiMasterOut, qkvOut, outputValid, readyForInput, head0Debug)
 where

  layerParams = modelLayers params !! layerIdx
  mhaParams = PARAM.multiHeadAttention layerParams

  xNorm = rmsNormFwFix <$> xVec <*> pure (PARAM.rmsAttF mhaParams)

  qResults = map (qHead params) indicesI
    where
      qHead :: PARAM.DecoderParameters
            -> Index NumQueryHeads
            -> ( Master.AxiMasterOut dom
              , Signal dom (Vec HeadDimension FixedPoint)
              , Signal dom Bool
              , Signal dom Bool
              , QHeadDebugInfo dom
              )
      qHead params' headIdx = queryHeadProjector dramSlaveIn layerIdx headIdx
                          inputValid downStreamReady seqPos xNorm params'

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
      in keyValueHeadProjector inputValid downStreamReady seqPos xNorm headParams'

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

  (axiMasterOut, result, matVecValid, projReadyOut, debugInfo) =
    qkvProjector dramSlaveIn layerIdx enableGated downStreamReady
                 seqPos input params

  projReadyOut_d = register True projReadyOut
  enableGated    = enableRaw  .&&. projReadyOut_d
  readyForInput  = inReadyRaw .&&. projReadyOut_d
