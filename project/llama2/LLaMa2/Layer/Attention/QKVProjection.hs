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
import qualified Prelude as P

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

data MultiplierDebug dom = MultiplierDebug 
  { accValue  :: Signal dom FixedPoint
  , rowReset  :: Signal dom Bool
  , rowEnable :: Signal dom Bool
  } deriving (Generic)

data MultiplierOutput dom = MultiplierOutput
  { moRowResult     :: Signal dom FixedPoint
  , moRowDone       :: Signal dom Bool
  , moState         :: Signal dom OPS.MultiplierState
  , moRowReqValid   :: Signal dom Bool
  , moOutputValid   :: Signal dom Bool
  , moReadyForInput :: Signal dom Bool
  , moDebug         :: MultiplierDebug dom
  } deriving (Generic)

multiplier :: forall dom. 
  HiddenClockResetEnable dom
  => Signal dom (Vec ModelDimension FixedPoint)
  -> Signal dom (RowI8E ModelDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
  -> MultiplierOutput dom
multiplier column row colValid rowValid downStreamReady rowIndex = 
  MultiplierOutput
    { moRowResult     = rowResult
    , moRowDone       = rowDone
    , moState         = state
    , moRowReqValid   = rowReqValid
    , moOutputValid   = outputValid
    , moReadyForInput = readyForInputRaw
    , moDebug         = dbgInfo
    }
  where
    (rowResult, rowDone, accValue) = 
      OPS.parallel64RowProcessor rowReset rowEnable row column
  
    (state, rowReqValid, rowReset, rowEnable, outputValid, readyForInputRaw) =
      OPS.matrixMultiplierStateMachine colValid rowValid downStreamReady rowDone rowIndex

    dbgInfo = MultiplierDebug 
      { accValue  = accValue
      , rowReset  = rowReset
      , rowEnable = rowEnable
      }

assertRowsMatch :: forall dom n m.
  (KnownNat n)
  => Signal dom Bool
  -> Signal dom (Index m)
  -> Signal dom (RowI8E n)     -- dramRow
  -> Signal dom (RowI8E n)     -- hcRow  
  -> Signal dom (RowI8E n)
assertRowsMatch guard _rowIdx dramRow hcRow = result
  where
    dramExp = rowExponent <$> dramRow
    hcExp   = rowExponent <$> hcRow
    
    expMatch = dramExp .==. hcExp
    
    -- Check specific negative values that indicate common bugs
    dramExpIs0    = dramExp .==. pure 0
    dramExpIsNeg1 = dramExp .==. pure (-1)   -- 0xFF - often uninitialized or all-ones
    dramExpIsNeg128 = dramExp .==. pure (-128) -- 0x80 - sign bit only
    dramExpGtNeg10 = dramExp .>. pure (-10)  -- Small negative
    dramExpGtNeg64 = dramExp .>. pure (-64)  -- Medium negative
    
    result = mux (guard .&&. (not <$> expMatch))
                 (mux dramExpIsNeg1
                      (errorX "DRAM exp = -1 (0xFF, all ones?)")
                      (mux dramExpIsNeg128
                           (errorX "DRAM exp = -128 (0x80, sign bit only)")
                           (mux dramExpGtNeg10
                                (errorX "DRAM exp in [-9, -1)")
                                (mux dramExpGtNeg64
                                     (errorX "DRAM exp in [-63, -10]")
                                     (errorX "DRAM exp in [-128, -64]")))))
                 dramRow

--------------------------------------------------------------------------------
-- High-level query head matrix multiplier with DRAM weight loading
--------------------------------------------------------------------------------
data QueryHeadOutput dom = QueryHeadOutput
  { qhoAxiMaster     :: Master.AxiMasterOut dom
  , qhoResult        :: Signal dom (Vec HeadDimension FixedPoint)
  , qhoOutputValid   :: Signal dom Bool
  , qhoReadyForInput :: Signal dom Bool
  , qhoDebugInfo     :: QHeadDebugInfo dom
  } deriving (Generic)

queryHeadMatrixMultiplier :: forall dom.
  HiddenClockResetEnable dom
  => Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.DecoderParameters
  -> QueryHeadOutput dom
queryHeadMatrixMultiplier dramSlaveIn layerIdx headIdx inputValid downStreamReady xHat params =
  QueryHeadOutput
    { qhoAxiMaster     = axiMaster
    , qhoResult        = qOut
    , qhoOutputValid   = outputValid
    , qhoReadyForInput = readyForInput
    , qhoDebugInfo     = debugInfo
    }
 where
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex
  
  -- Weight loader
  (axiMaster, weightLoaderOut, weightValid, weightReady) = 
    LOADER.weightLoader dramSlaveIn layerIdx headIdx 
                 rowIndex rowReqValidGated downStreamReady params
  
  -- Select weights (hardcoded or DRAM)
  currentRow = LOADER.dramRowOut weightLoaderOut
  currentRow' = LOADER.hcRowOut weightLoaderOut

  -- THE ASSERTION: check when multiplier is actively processing
  rowBeingUsed = rowEnable (moDebug multOut)
  
  -- Route through the assertion - now it's in the actual datapath
  checkedRow = assertRowsMatch rowBeingUsed rowIndex currentRow currentRow'

  -- Matrix multiplier with clean interface
  multOut = multiplier xHat checkedRow inputValid weightValid downStreamReady rowIndex

  -- Gate signals with weightReady
  rowReqValidGated = moRowReqValid multOut .&&. weightReady
  readyForInput = moReadyForInput multOut .&&. weightReady
  outputValid = moOutputValid multOut
  
  -- Row index management
  nextRowIndex = mux (moRowDone multOut .&&. (rowIndex ./=. pure maxBound))
                     (rowIndex + 1)
                     (mux ((moState multOut .==. pure OPS.MDone) .&&. downStreamReady)
                          (pure 0)
                          rowIndex)
  
  -- Accumulate results
  qOut = register (repeat 0) nextOutput
  nextOutput = mux (moRowDone multOut)
                   (replace <$> rowIndex <*> moRowResult multOut <*> qOut)
                   qOut
  
  -- Debug info
  debugInfo = QHeadDebugInfo
    { qhRowIndex        = rowIndex
    , qhState           = moState multOut
    , qhFirstMant       = register 0 (head . rowMantissas <$> currentRow')
    , qhRowResult       = register 0 (moRowResult multOut)
    , qhRowDone         = moRowDone multOut
    , qhFetchValid      = weightValid
    , qhFetchedWord     = pure 0
    , qhRowReset        = rowReset (moDebug multOut)
    , qhRowEnable       = rowEnable (moDebug multOut)
    , qhAccumValue      = accValue (moDebug multOut)
    , qhQOut            = qOut
    , qhCurrentRowExp   = register 0 (rowExponent <$> currentRow)
    , qhCurrentRowMant0 = register 0 (head . rowMantissas <$> currentRow)
    , qhRowReqValid     = moRowReqValid multOut
    , qhWeightReady     = weightReady
    , qhWeightValid     = weightValid
    }

--------------------------------------------------------------------------------
-- Q head projector (now trivially simple!)
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
  (qhoAxiMaster qhOut, qRoOut, qhoOutputValid qhOut, qhoReadyForInput qhOut, qhoDebugInfo qhOut)
 where
  qhOut = queryHeadMatrixMultiplier dramSlaveIn layerIdx headIdx 
                                    inputValid downStreamReady xHat params
  
  qRoOut = (rotaryEncoder (PARAM.rotaryF (PARAM.headsQ (PARAM.multiHeadAttention (modelLayers params !! layerIdx)) !! headIdx)) <$> stepCount) <*> qhoResult qhOut

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
