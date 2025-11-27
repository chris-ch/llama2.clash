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
import Simulation.Parameters (DecoderParameters(..))
import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER

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
  , qhFetchedWord :: Signal dom (BitVector 512)
  , qhRowReset     :: Signal dom Bool
  , qhRowEnable    :: Signal dom Bool
  , qhAccumValue   :: Signal dom FixedPoint
  , qhQOut         :: Signal dom (Vec HeadDimension FixedPoint)
  , qhCurrentRowExp    :: Signal dom Exponent
  , qhCurrentRowMant0  :: Signal dom Mantissa
  , qhRowReqValid      :: Signal dom Bool      -- State machine's request signal
  , qhWeightReady      :: Signal dom Bool      -- WeightLoader's ready signal
  , qhWeightValid      :: Signal dom Bool      -- WeightLoader's valid signal
  } deriving (Generic)

--------------------------------------------------------------------------------
-- Q head projector
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
queryHeadProjector dramSlaveIn layerIdx headIdx inputValid downStreamReady stepCount xHat params =
  (axiMaster, qRoOut, outputValid, readyForInput, debugInfo)
 where
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex
  
  -- Use nextRowIndex to drive the loader so address aligns with the valid pulse
  (axiMaster, weightLoaderOut, weightValid, weightReady) = 
    LOADER.weightLoader dramSlaveIn layerIdx headIdx 
                 rowIndex rowReqValidGated downStreamReady params
  
  -- MANUALLY select the weights type here
  currentRow = LOADER.dramRowOut weightLoaderOut
  currentRow' = LOADER.hcRowOut weightLoaderOut

  -- Processing with gated enable
  (rowResult, rowDone, colIdx, accValue) = 
    OPS.parallel64RowProcessor rowReset rowEnable currentRow' xHat
  
  (state, rowReqValid, rowReset, rowEnable, outputValid, readyForInputRaw) =
    OPS.matrixMultiplierStateMachine inputValid downStreamReady rowDone weightValid rowIndex
  
  -- Gate rowReqValid with weightReady**
  rowReqValidGated = rowReqValid .&&. weightReady

  readyForInput = readyForInputRaw .&&. weightReady
  
  nextRowIndex = mux (rowDone .&&. (rowIndex ./=. pure maxBound))
                     (rowIndex + 1)
                     (mux ((state .==. pure OPS.MDone) .&&. downStreamReady)
                          (pure 0)
                          rowIndex)
  
  qOut = register (repeat 0) nextOutput
  nextOutput = mux rowDone
                   (replace <$> rowIndex <*> rowResult <*> qOut)
                   qOut
  
  qRoOut = (rotaryEncoder (PARAM.rotaryF (PARAM.headsQ (PARAM.multiHeadAttention (modelLayers params !! layerIdx)) !! headIdx)) <$> stepCount) <*> qOut
  
  debugInfo = QHeadDebugInfo
    { qhRowIndex     = rowIndex
    , qhState        = state
    , qhFirstMant    = register 0 (head . rowMantissas <$> currentRow)
    , qhRowResult    = register 0 rowResult
    , qhRowDone      = rowDone
    , qhFetchValid   = weightValid
    , qhFetchedWord  = pure 0
    , qhRowReset     = rowReset
    , qhRowEnable    = rowEnable
    , qhAccumValue   = accValue
    , qhQOut         = qOut
    , qhCurrentRowExp    = register 0 (rowExponent <$> currentRow)
    , qhCurrentRowMant0  = register 0 (head . rowMantissas <$> currentRow)
    , qhRowReqValid      = rowReqValid
    , qhWeightReady      = weightReady
    , qhWeightValid      = weightValid
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
