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
import LLaMa2.Memory.AXI.Types (AxiR(..), AxiB (..), AxiAW (..), AxiW (..))
import Clash.Debug (trace)

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
    rowValidRise = rowValid .&&. (not <$> register False rowValid)

    colValidTraced = go <$> rowValidRise <*> colValid
      where
        go True cv = trace ("MULT: rowValid ROSE, colValid=" P.++ show cv) cv
        go False cv = cv

    (rowResult, rowDone, accValue) =
      OPS.parallel64RowProcessor rowReset rowEnable row column

    (state, rowReqValid, rowReset, rowEnable, outputValid, readyForInputRaw) =
      OPS.matrixMultiplierStateMachine colValidTraced rowValid downStreamReady rowDone rowIndex

    dbgInfo = MultiplierDebug
      { accValue  = accValue
      , rowReset  = rowReset
      , rowEnable = rowEnable
      }

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
    , qhoResult        = qOutFinal
    , qhoOutputValid   = outputValid
    , qhoReadyForInput = readyForInput
    , qhoDebugInfo     = debugInfo
    }
 where
  rowIndex :: Signal dom (Index HeadDimension)
  rowIndex = register 0 nextRowIndex

  -- Latch inputValid until we complete all rows
  inputValidLatched :: Signal dom Bool
  inputValidLatched = register False nextInputValidLatched
    where
      nextInputValidLatched = mux inputValid (pure True)
                            $ mux (outputValid .&&. downStreamReady) (pure False)
                              inputValidLatched

  weightValidRise = weightValid .&&. (not <$> register False weightValid)

  inputValidLatched' = go <$> weightValidRise <*> inputValidLatched
    where
      go True ivl = trace ("H" P.++ show headIdx P.++ " WVALID_RISE ivl=" P.++ show ivl) ivl
      go False ivl = ivl

  -- Weight loader (note: pass moRowDone to allow the loader to hold LDone until consumed)
  (axiMaster, weightLoaderOut, weightValid, weightReady) =
    LOADER.weightLoader dramSlaveIn layerIdx headIdx
                        rowIndex rowReqValidTraced downStreamReady
                        (moRowDone multOut)
                        params

  -- COMMITTED rows from loader
  currentRowDramRaw = LOADER.dramRowOut weightLoaderOut
  currentRowHCRaw   = LOADER.hcRowOut   weightLoaderOut

  -- Ensure rows don't change while valid (live-path assertion)
  currentRowDram = LOADER.assertRowStable weightValid currentRowDramRaw
  currentRowHC   = LOADER.assertRowStable weightValid currentRowHCRaw

  -- DRAM path multiplier
  multOut = multiplier xHat currentRowDram inputValidLatched' weightValid downStreamReady rowIndex

  -- HC path: reuse the DRAM path's control signals
  (hcRowResult, _hcRowDone, _hcAccValue) =
    OPS.parallel64RowProcessor
      (rowReset (moDebug multOut))
      (rowEnable (moDebug multOut))
      currentRowHC
      xHat

  -- Assert row results match exactly when rowDone fires; feed the checked DRAM result forward
  dramRowResultChecked :: Signal dom FixedPoint
  dramRowResultChecked = assertRowResultMatch
                           (moRowDone multOut)
                           rowIndex
                           (moRowResult multOut)
                           hcRowResult
                           currentRowDram   -- use committed+stable rows for debug
                           currentRowHC

  rowReqValidGated = moRowReqValid multOut .&&. weightReady

  traceInputValid :: Signal dom Bool -> Signal dom Bool
  traceInputValid sig = go <$> sig <*> inputValid <*> weightValid <*> rowIndex
    where
      go req True wv ri = trace ("H" P.++ show headIdx P.++ " INPUT_VALID wv=" P.++ show wv P.++ " ri=" P.++ show ri) req
      go req False _ _ = req

  rowReqValidTraced = traceInputValid rowReqValidGated

  readyForInput    = moReadyForInput multOut .&&. weightReady

    -- Latch outputValid until downstream consumes it
  outputValidLatch :: Signal dom Bool
  outputValidLatch = register False nextOutputValidLatch
    where
      -- Set when multiplier signals done, clear when downstream ready AND we're valid
      nextOutputValidLatch = mux (moOutputValid multOut) (pure True)
                          $ mux (outputValidLatch .&&. downStreamReady) (pure False)
                            outputValidLatch

  -- Use the latched version externally
  outputValid = outputValidLatch

  nextRowIndex =
    mux (moRowDone multOut .&&. (rowIndex ./=. pure maxBound))
        (rowIndex + 1)
        (mux (outputValidLatch .&&. downStreamReady)  -- Only reset when latch clears
             (pure 0)
             rowIndex)

  -- Accumulate using checked DRAM result
  qOut = register (repeat 0) nextOutput
  nextOutput = mux (moRowDone multOut)
                   (replace <$> rowIndex <*> dramRowResultChecked <*> qOut) -- ! dramRowResultChecked, use hcRowResult to disable comparison
                   qOut

  -- Accumulate HC results (reference)
  qOutHC :: Signal dom (Vec HeadDimension FixedPoint)
  qOutHC = register (repeat 0) nextOutputHC
  nextOutputHC = mux (moRowDone multOut)
                     (replace <$> rowIndex <*> hcRowResult <*> qOutHC)
                     qOutHC

  qOutChecked = assertQOutputsMatch outputValid rowIndex qOut qOutHC

  traceMultState :: Signal dom (Vec HeadDimension FixedPoint) -> Signal dom (Vec HeadDimension FixedPoint)
  traceMultState qOut' = go <$> moState multOut <*> outputValid <*> rowIndex <*> moRowDone multOut <*> downStreamReady <*> qOut'
    where
      go st ov ri rd dsr out =
        let msg = "H" P.++ show headIdx
                  P.++ " st=" P.++ show st
                  P.++ " ov=" P.++ show ov
                  P.++ " ri=" P.++ show ri
                  P.++ " rd=" P.++ show rd
                  P.++ " dsr=" P.++ show dsr
        in if rd  -- Trace every time rowDone fires
          then trace msg out
          else out

  qOutFinal = traceMultState qOutChecked

  debugInfo = QHeadDebugInfo
    { qhRowIndex        = rowIndex
    , qhState           = moState multOut
    , qhFirstMant       = register 0 (head . rowMantissas <$> currentRowHC)
    , qhRowResult       = register 0 (moRowResult multOut)
    , qhRowDone         = moRowDone multOut
    , qhFetchValid      = weightValid
    , qhFetchedWord     = pure 0
    , qhRowReset        = rowReset (moDebug multOut)
    , qhRowEnable       = rowEnable (moDebug multOut)
    , qhAccumValue      = accValue (moDebug multOut)
    , qhQOut            = qOut
    , qhCurrentRowExp   = register 0 (rowExponent <$> currentRowDram)
    , qhCurrentRowMant0 = register 0 (head . rowMantissas <$> currentRowDram)
    , qhRowReqValid     = moRowReqValid multOut
    , qhWeightReady     = weightReady
    , qhWeightValid     = weightValid
    }

-- | Assert that row results match when rowDone fires, with detailed debug
assertRowResultMatch :: forall dom . HiddenClockResetEnable dom
  => Signal dom Bool                    -- ^ rowDone trigger
  -> Signal dom (Index HeadDimension)   -- ^ row index
  -> Signal dom FixedPoint              -- ^ DRAM result
  -> Signal dom FixedPoint              -- ^ HC result
  -> Signal dom (RowI8E ModelDimension) -- ^ DRAM weights (for debug)
  -> Signal dom (RowI8E ModelDimension) -- ^ HC weights (for debug)
  -> Signal dom FixedPoint
assertRowResultMatch rowDone rowIdx dramResult hcResult dramWeights hcWeights = result
  where
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0 nextTokenCnt
    nextTokenCnt = mux (rowDone .&&. (rowIdx .==. pure maxBound))
                       (tokenCnt + 1)
                       tokenCnt

    result = mux rowDone
                 (check <$> tokenCnt <*> rowIdx <*> dramResult <*> hcResult 
                        <*> dramWeights <*> hcWeights)
                 dramResult

    check :: Unsigned 32 -> Index HeadDimension -> FixedPoint -> FixedPoint 
          -> RowI8E ModelDimension -> RowI8E ModelDimension -> FixedPoint
    check tok ri dr hr dramW hcW =
      if dr P.== hr
        then dr
        else P.error $ "Row result mismatch at token " P.++ show tok 
                    P.++ " row " P.++ show ri
                    P.++ ": DRAM=" P.++ show dr 
                    P.++ " HC=" P.++ show hr
                    P.++ "\n  DRAM weight exp=" P.++ show (rowExponent dramW)
                    P.++ " mant[0]=" P.++ show (P.head (toList (rowMantissas dramW)))
                    P.++ "\n  HC weight exp=" P.++ show (rowExponent hcW)
                    P.++ " mant[0]=" P.++ show (P.head (toList (rowMantissas hcW)))
                    P.++ "\n  weights match=" P.++ show (rowExponent dramW P.== rowExponent hcW 
                                                         P.&& rowMantissas dramW P.== rowMantissas hcW)

-- | Compare DRAM and HC Q outputs when valid - X-safe version
assertQOutputsMatch
  :: forall dom n. (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom Bool                      -- ^ outputValid (from DRAM path)
  -> Signal dom (Index HeadDimension)
  -> Signal dom (Vec n FixedPoint)         -- ^ DRAM result (real)
  -> Signal dom (Vec n FixedPoint)         -- ^ HC result (reference)
  -> Signal dom (Vec n FixedPoint)
assertQOutputsMatch outputValid _rowIdx dramOut hcOut = result
 where
  -- Detect the first valid output to skip initial undefined/warm-up phase
  everValid :: Signal dom Bool
  everValid = register False (everValid .||. outputValid)

  -- one-cycle delayed view of the outputValid
  prevOutputValid :: Signal dom Bool
  prevOutputValid = register False outputValid

  -- Trigger comparison one cycle *after* outputValid goes high
  -- (i.e. when prevOutputValid is True)
  checkTrigger :: Signal dom Bool
  checkTrigger = prevOutputValid .&&. everValid

  -- Sample only when we are sure data is valid and defined
  dramSampled = register (repeat 0) (mux checkTrigger dramOut dramSampled)
  hcSampled   = register (repeat 0) (mux checkTrigger hcOut   hcSampled)

  -- Simple, correct counter
  tokenCnt :: Signal dom (Unsigned 32)
  tokenCnt = register 0 (mux checkTrigger (tokenCnt + 1) tokenCnt)

  -- Final output: substitute checked value only when checkTrigger fires
  result = mux checkTrigger (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled) dramOut

  -- Pure function — safe, uses only Prelude, no Clash (==) on undefined BitVectors
  checkPure :: Unsigned 32 -> Vec n FixedPoint -> Vec n FixedPoint -> Vec n FixedPoint
  checkPure tok dr hr =
    let ds = toList dr
        hs = toList hr
        pairs = P.zip [0..] (P.zip ds hs)
        mismatches = P.filter (\(_, (d,h)) -> d P./= h) pairs
    in if P.null mismatches
       then dr
       else let (i, (d, h)) = P.head mismatches
            in P.error $ "QHead output mismatch at token " P.++ show tok P.++
                         ": first mismatch at index " P.++ show (i :: Int) P.++
                         " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")" P.++
                         " [total mismatches: " P.++ show (P.length mismatches) P.++ "]"

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
  (qhoAxiMaster qhOut, qRoOut, qhoOutputValid qhOut, qhoReadyForInput qhOut, qhoDebugInfo qhOut)
 where
  qhOut = queryHeadMatrixMultiplier dramSlaveIn layerIdx headIdx
                                    inputValid downStreamReady xHat params

  qRoOut = (rotaryEncoder (PARAM.rotaryEncoding params) <$> stepCount) <*> qhoResult qhOut

--------------------------------------------------------------------------------
-- KV head projector
--------------------------------------------------------------------------------
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> PARAM.KeyValueHeadComponentQ
  -> PARAM.RotaryEncodingComponentF
  -> ( Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     )
keyValueHeadProjector inputValid downStreamReady stepCountSig xHatSig kvHeadParams rotary =
  (kRoOut, vOut, outputValid, readyForInput)
 where
  selectedK :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedK = pure (PARAM.kMatrix kvHeadParams)

  selectedV :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedV = pure (PARAM.vMatrix kvHeadParams)

  (kOut, kValidOut, kReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedV xHatSig

  kRoOut = (rotaryEncoder rotary <$> stepCountSig) <*> kOut

  outputValid = kValidOut .&&. vValidOut
  readyForInput = kReadyOut .&&. vReadyOut

-- | AXI arbiter that properly routes responses back to the requesting master
axiArbiterWithRouting :: forall dom n.
  (HiddenClockResetEnable dom, KnownNat n)
  => Slave.AxiSlaveIn dom              -- ^ Single DRAM slave
  -> Vec n (Master.AxiMasterOut dom)   -- ^ Multiple masters (heads)
  -> ( Master.AxiMasterOut dom         -- ^ Combined master to DRAM
     , Vec n (Slave.AxiSlaveIn dom)    -- ^ Per-head slave interfaces
     )
axiArbiterWithRouting slaveIn masters = (masterOut, perHeadSlaves)
  where
    arRequests :: Vec n (Signal dom Bool)
    arRequests = map Master.arvalid masters

    -- Transaction tracking state machine
    inFlight :: Signal dom Bool
    inFlight = register False nextInFlight

    transactionOwner :: Signal dom (Index n)
    transactionOwner = register 0 nextTransactionOwner

    lastGranted :: Signal dom (Index n)
    lastGranted = register 0 nextLastGranted

    -- Round-robin selection: find next requesting head
    nextRequester :: Signal dom (Index n)
    nextRequester = findNextRequester <$> bundle arRequests <*> lastGranted

    findNextRequester :: Vec n Bool -> Index n -> Index n
    findNextRequester reqs lastR =
      let start = if lastR == maxBound then 0 else lastR + 1
          go i cnt
            | cnt == (0 :: Int) = lastR
            | reqs !! i = i
            | i == maxBound = go 0 (cnt - 1)
            | otherwise = go (i + 1) (cnt - 1)
      in go start (natToNum @n)

    -- Active index: locked to owner when in-flight, otherwise round-robin
    activeIdx :: Signal dom (Index n)
    activeIdx = mux inFlight transactionOwner nextRequester

    -- AR handshake detection
    selectedArValid = (!!) <$> bundle arRequests <*> activeIdx
    arHandshake = selectedArValid .&&. Slave.arready slaveIn .&&. (not <$> inFlight)

    -- R channel handshake detection
    selectedRReady = (!!) <$> bundle (map Master.rready masters) <*> transactionOwner
    rHandshake = Slave.rvalid slaveIn .&&. selectedRReady
    rLast = rlast <$> Slave.rdata slaveIn
    transactionDone = rHandshake .&&. rLast .&&. inFlight

    -- State transitions
    nextInFlight = mux arHandshake (pure True)
                 $ mux transactionDone (pure False)
                   inFlight

    nextTransactionOwner = mux arHandshake activeIdx transactionOwner

    -- Update lastGranted only when a transaction completes (for fair round-robin)
    nextLastGranted = mux transactionDone transactionOwner lastGranted

    -- Build master output using activeIdx for AR, transactionOwner for R
    masterOut = Master.AxiMasterOut
      { arvalid = mux inFlight (pure False) selectedArValid  -- Don't issue AR while in-flight
      , ardata  = (!!) <$> bundle (map Master.ardata masters) <*> activeIdx
      , rready  = (!!) <$> bundle (map Master.rready masters) <*> transactionOwner
      , awvalid = pure False
      , awdata  = pure (AxiAW 0 0 0 0 0)
      , wvalid  = pure False
      , wdata   = pure (AxiW 0 0 False)
      , bready  = pure False
      }

    -- Per-head slave interfaces with response routing
    perHeadSlaves :: Vec n (Slave.AxiSlaveIn dom)
    perHeadSlaves = map makeHeadSlave indicesI

    makeHeadSlave :: Index n -> Slave.AxiSlaveIn dom
    makeHeadSlave headIdx = Slave.AxiSlaveIn
      { arready = isActiveAndIdle .&&. Slave.arready slaveIn
      , rvalid  = isOwner .&&. Slave.rvalid slaveIn
      , rdata   = Slave.rdata slaveIn
      , awready = pure False
      , wready  = pure False
      , bvalid  = pure False
      , bdata   = pure (AxiB 0 0)
      }
      where
        isActiveAndIdle = (activeIdx .==. pure headIdx) .&&. (not <$> inFlight)
        isOwner = inFlight .&&. (transactionOwner .==. pure headIdx)

--------------------------------------------------------------------------------
-- QKV projector
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

  -- Get global rotary once
  rotary = PARAM.rotaryEncoding params

  -- Create heads with per-head routed slave interfaces
  qResults :: Vec NumQueryHeads (Master.AxiMasterOut dom, Signal dom (Vec HeadDimension FixedPoint), Signal dom Bool, Signal dom Bool, QHeadDebugInfo dom)
  qAxiMasters :: Vec NumQueryHeads (Master.AxiMasterOut dom)
  perHeadSlaves :: Vec NumQueryHeads (Slave.AxiSlaveIn dom)
  
  (axiMasterOut, perHeadSlaves) = axiArbiterWithRouting dramSlaveIn qAxiMasters

  qResults = map (qHead params) indicesI
    where
      qHead params' headIdx = queryHeadProjector dramSlaveIn layerIdx headIdx
                        inputValid downStreamReady seqPos xNorm params'

  --- IDEALLY IT SHOULD BE THE BELOW BUT IT BREAKS THE CONSTANT PARAMETERS PATH
  consumeSignal = outputValid .&&. downStreamReady

  qResults' = imap (\headIdx _ ->
      queryHeadProjector (perHeadSlaves !! headIdx) layerIdx headIdx
                        inputValid consumeSignal seqPos xNorm params
    ) (repeat () :: Vec NumQueryHeads ())

  head0Debug = head qDebugInfos
  qAxiMasters = map (\(axi, _, _, _, _) -> axi) qResults
  qVecs       = map (\(_, q, _, _, _) -> q) qResults
  qValids     = map (\(_, _, v, _, _) -> v) qResults
  qReadys     = map (\(_, _, _, r, _) -> r) qResults
  qDebugInfos = map (\(_, _, _, _, d) -> d) qResults

  kvResults = map kvHead indicesI
   where
    kvHead kvIdx =
      let kvHeadParams = PARAM.kvHeads mhaParams !! kvIdx  -- Get actual KV head
      in keyValueHeadProjector inputValid downStreamReady seqPos xNorm kvHeadParams rotary
  
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
