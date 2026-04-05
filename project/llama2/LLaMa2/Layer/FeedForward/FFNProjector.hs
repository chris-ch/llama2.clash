module LLaMa2.Layer.FeedForward.FFNProjector
  ( ffnProjector
  ) where

import Clash.Prelude
import qualified GHC.TypeNats as TN

import LLaMa2.Types.ModelConfig
    ( ModelDimension, HiddenDimension, NumLayers, NumQueryHeads )
import LLaMa2.Numeric.Types (FixedPoint, scalePow2F)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Numeric.Operations (MultiplierState, matrixMultiplierStateMachine)
import LLaMa2.Layer.FeedForward.Activation (sigmoidLinearUnit)
import LLaMa2.Memory.DualPortRAM (trueDualPortRam)

import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Arbiter as ARB

import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OTC
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController  as ITC
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler                as RS

--------------------------------------------------------------------------------
-- FFN intermediate BRAM address types
--
-- Slot A [0 .. HiddenDim-1]           : w1 (gate) results
-- Slot B [HiddenDim .. 2*HiddenDim-1] : SiLU(gate)*up product (w2 column input)
-- Slot C [2*HiddenDim .. 2*HiddenDim+ModelDim-1] : w2 (down) results
--------------------------------------------------------------------------------
type FFNBramDepth = 2 TN.* HiddenDimension TN.+ ModelDimension
type FFNBramAddr  = Index FFNBramDepth

ffnSlotBBase :: FFNBramAddr
ffnSlotBBase = natToNum @HiddenDimension

ffnSlotCBase :: FFNBramAddr
ffnSlotCBase = natToNum @(2 TN.* HiddenDimension)

--------------------------------------------------------------------------------
-- FFN Phase FSM
--------------------------------------------------------------------------------

data FFNProjState = FPIdle | FPGate | FPUp | FPDown | FPDone
  deriving (Show, Eq, Generic, NFDataX)

--------------------------------------------------------------------------------
-- Row request pulse helper
--------------------------------------------------------------------------------

mkRowReqPulse :: forall dom numRows.
  ( HiddenClockResetEnable dom, KnownNat numRows )
  => Signal dom (Unsigned 32)
  -> Signal dom Bool              -- ^ rcFetchReq from RowComputeUnit
  -> Signal dom Bool              -- ^ weightReady
  -> Signal dom (Index numRows)   -- ^ effectiveRowIndex
  -> Signal dom Bool
mkRowReqPulse _cycleCounter fetchReq weightReady effRowIdx = pulse
  where
    loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)
    reqValidGated    = fetchReq .&&. weightReady
    prevReqValid     = register False $ mux loaderBecameIdle (pure False) reqValidGated
    reqRise          = reqValidGated .&&. (not <$> prevReqValid)
    prevRowIdx       = register (0 :: Index numRows) effRowIdx
    rowIdxChanged    = effRowIdx ./=. prevRowIdx
    pulse            = reqRise .||. (reqValidGated .&&. rowIdxChanged)

--------------------------------------------------------------------------------
-- Inline serial row accumulator
--
-- Mirrors the W2 inline pattern: one element per cycle, BRAM-fed column.
-- Returns (state, fetchReq, rowReset, rowEnable, allDone, idleReady,
--          compCounter, rowDone, serialResult)
-- where serialResult is valid when rowDone fires.
--------------------------------------------------------------------------------

serialRowAccum :: forall dom numRows numCols.
  ( HiddenClockResetEnable dom
  , KnownNat numRows
  , KnownNat numCols
  , 1 <= numCols
  )
  => Signal dom Bool                    -- ^ effInput  (colValid)
  -> Signal dom Bool                    -- ^ weightValid (rowValid)
  -> Signal dom Bool                    -- ^ rowDoneIn  (rowDone — fed back from output)
  -> Signal dom (Index numRows)         -- ^ rowIndex
  -> Signal dom (RowI8E numCols)        -- ^ weightRow (from DRAM)
  -> Signal dom FixedPoint              -- ^ colRdData (from BRAM, 1-cycle latency)
  -> ( Signal dom MultiplierState
     , Signal dom Bool   -- fetchReq
     , Signal dom Bool   -- rowReset
     , Signal dom Bool   -- rowEnable
     , Signal dom Bool   -- allDone
     , Signal dom Bool   -- idleReady
     , Signal dom (Index numCols)  -- compCounter (column element index)
     , Signal dom Bool             -- rowDone
     , Signal dom FixedPoint       -- serialResult (valid when rowDone fires)
     , Signal dom (Index numCols)  -- colPrefetch (pre-fetch address for column BRAM)
     )
serialRowAccum effInput weightValid rowDoneIn rowIndex weightRow colRdData =
  (machState, fetchReq, rowReset, rowEnable, allDone, idleReady,
   compCounter, rowDone, serialResult, colPrefetch)
  where
    (machState, fetchReq, rowReset, rowEnable, allDone, idleReady) =
      matrixMultiplierStateMachine
        effInput weightValid (pure True) rowDoneIn rowIndex

    -- Column element counter: reset on MReset, advance on MProcessing
    compCounter :: Signal dom (Index numCols)
    compCounter = register 0 nextCompCounter

    nextCompCounter :: Signal dom (Index numCols)
    nextCompCounter =
      mux rowReset   (pure 0) $
      mux rowEnable  (satSucc SatBound <$> compCounter) $
      compCounter

    -- Pre-fetch address: issued 1 cycle ahead so data arrives on time
    -- MReset  → 0 (delivers col[0] at first MProcessing)
    -- MProcess k → k+1 (delivers col[k+1] next cycle)
    colPrefetch :: Signal dom (Index numCols)
    colPrefetch =
      mux rowReset (pure 0) (satSucc SatBound <$> compCounter)

    -- Serial multiply-accumulate
    mantissaElem :: Signal dom (Signed 8)
    mantissaElem = (!!) <$> (rowMantissas <$> weightRow) <*> compCounter

    product' :: Signal dom FixedPoint
    product' = (fromIntegral <$> mantissaElem) * colRdData

    acc :: Signal dom FixedPoint
    acc = register 0 nextAcc

    nextAcc :: Signal dom FixedPoint
    nextAcc =
      mux rowReset (pure 0) $
      mux (rowEnable .&&. (not <$> rowDoneIn)) (acc + product') $
      acc

    -- Row done detection: rising edge of lastElemFlag, then registered
    lastElemFlag :: Signal dom Bool
    lastElemFlag = (compCounter .==. pure maxBound) .&&. rowEnable

    rowDoneRaw :: Signal dom Bool
    rowDoneRaw = lastElemFlag .&&. (not <$> register False lastElemFlag)

    rowDone :: Signal dom Bool
    rowDone = register False rowDoneRaw

    -- Scale by quantisation exponent to produce the dot-product result
    serialResult :: Signal dom FixedPoint
    serialResult = scalePow2F <$> (rowExponent <$> weightRow) <*> acc

--------------------------------------------------------------------------------
-- ffnProjector
--
-- DRAM-backed FFN with BRAM-backed intermediate storage.
-- Sequential phases:
--   FPGate: W1 (gate)   — HiddenDimension rows × ModelDimension cols
--           Column (xHat) read from xHat BRAM, results written to slot A.
--   FPUp:   W3 (up)     — HiddenDimension rows × ModelDimension cols
--           Column (xHat) read from xHat BRAM, SiLU(gate)*up to slot B.
--   FPDown: W2 (down)   — ModelDimension rows × HiddenDimension cols
--           Column (slot B) read from FFN BRAM serially.
--           Results written to slot C element-by-element.
--------------------------------------------------------------------------------

ffnProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Signal dom Bool                                         -- ^ validIn
  -> Signal dom Bool                                         -- ^ readyIn
  -> Signal dom (Maybe (Index ModelDimension, FixedPoint))   -- ^ xHatWrite (from rmsNormSeq)
  -> Signal dom (Index ModelDimension)                       -- ^ ffnCRdAddr: slot C read addr (from caller)
  -> ( Master.AxiMasterOut dom
     , Signal dom FixedPoint                      -- ^ FFN BRAM read data (slot C, for residual FSM)
     , Signal dom Bool                            -- ^ validOut
     , Signal dom Bool                            -- ^ readyOut
     )
ffnProjector cycleCounter dramSlaveIn layerIdx validIn readyIn xHatWrite ffnCRdAddr =
  (axiMasterOut, ffnBramRdData, validOut, readyOut)
 where
  headIdx = 0 :: Index NumQueryHeads

  -------------------------------------------------------------------------
  -- 3-master AXI sub-arbiter: slot 0 = W1, slot 1 = W3, slot 2 = W2
  -------------------------------------------------------------------------
  (axiMasterOut, perPhaseSlaves) =
    ARB.axiArbiterWithRouting dramSlaveIn
      (w1AxiMaster :> w3AxiMaster :> w2AxiMaster :> Nil)

  w1Slave = perPhaseSlaves !! (0 :: Index 3)
  w3Slave = perPhaseSlaves !! (1 :: Index 3)
  w2Slave = perPhaseSlaves !! (2 :: Index 3)

  -------------------------------------------------------------------------
  -- Phase FSM
  -------------------------------------------------------------------------
  fpState :: Signal dom FFNProjState
  fpState = register FPIdle fpNextState

  fpNextState :: Signal dom FFNProjState
  fpNextState =
    mux acceptInput                                           (pure FPGate)
    $ mux (fpState .==. pure FPGate .&&. w1OutputValid)      (pure FPUp)
    $ mux (fpState .==. pure FPUp   .&&. w3OutputValid)      (pure FPDown)
    $ mux (fpState .==. pure FPDown .&&. w2OutputValid)      (pure FPDone)
    $ mux (fpState .==. pure FPDone .&&. readyIn)            (pure FPIdle)
      fpState

  acceptInput = fpState .==. pure FPIdle .&&. validIn .&&. w1WeightReady

  -------------------------------------------------------------------------
  -- xHat BRAM
  -- Port A: read (W1 during FPGate, W3 during FPUp)
  -- Port B: write (xHatWrite from rmsNormSeq, happens before FPGate starts)
  --
  -- Read address mux: during FPGate use W1 pre-fetch, during FPUp use W3.
  -- Writes and reads are strictly sequential (RNNormalize ends before FPGate).
  -------------------------------------------------------------------------
  xHatBramRdAddr :: Signal dom (Index ModelDimension)
  xHatBramRdAddr =
    mux (fpState .==. pure FPGate) w1ColPrefetch $
    mux (fpState .==. pure FPUp)   w3ColPrefetch $
    pure 0

  (xHatBramRdData, _) = trueDualPortRam
    xHatBramRdAddr
    (pure Nothing)                         -- port A: read-only
    (maybe 0 fst <$> xHatWrite)            -- port B: dummy read addr
    xHatWrite                              -- port B: write when Just

  -------------------------------------------------------------------------
  -- FFN intermediate BRAM
  --   Port A: read  (slot A during FPUp, slot B during FPDown, slot C during FPDone)
  --   Port B: write (slot A during FPGate, slot B during FPUp, slot C during FPDown)
  -------------------------------------------------------------------------
  ffnBramRdData :: Signal dom FixedPoint
  ffnBramRdData = fst $ trueDualPortRam
    ffnBramRdAddr
    (pure Nothing)
    ffnBramWrAddr
    ffnBramWriteOp

  ffnBramWriteOp :: Signal dom (Maybe (FFNBramAddr, FixedPoint))
  ffnBramWriteOp =
    mux w1RowDone w1BramWriteOp $
    mux w3WriteEnabled w3SiluBramWriteOp $
    w2BramWriteOp

  ffnBramWrAddr :: Signal dom FFNBramAddr
  ffnBramWrAddr = maybe 0 fst <$> ffnBramWriteOp

  -- Read address mux:
  --   FPDown  → slot B + W2 column counter pre-fetch
  --   FPDone  → slot C + ffnCRdAddr (caller reads W2 results for residual)
  --   Other   → slot A + w3RowIndex (FPGate/FPUp: reads gate for SiLU)
  ffnBramRdAddr :: Signal dom FFNBramAddr
  ffnBramRdAddr =
    mux (fpState .==. pure FPDown)
      ((ffnSlotBBase +) . fromIntegral <$> w2BramPrefetch)
    $ mux (fpState .==. pure FPDone)
      ((ffnSlotCBase +) . fromIntegral <$> ffnCRdAddr)
    $ fromIntegral <$> w3RowIndex

  -------------------------------------------------------------------------
  -- W1 (Gate) Phase — HiddenDimension rows × ModelDimension cols
  -------------------------------------------------------------------------
  w1RowIndex :: Signal dom (Index HiddenDimension)
  w1RowIndex = register 0 (RS.rsNextRowIndex w1RS)

  w1RS = RS.rowScheduler RS.RowSchedulerIn
    { RS.rsRowDone       = w1RowDone
    , RS.rsOutputValid   = w1OutputValid
    , RS.rsConsumeSignal = w1ConsumeSignal
    , RS.rsCurrentIndex  = w1RowIndex
    }

  w1EffRow :: Signal dom (Index HiddenDimension)
  w1EffRow = mux (w1OutputValid .&&. w1ConsumeSignal) (pure 0) w1RowIndex

  w1InputTxn = ITC.inputTransactionController cycleCounter headIdx
    ITC.InputTransactionIn
      { ITC.itcInputValid      = fpState .==. pure FPGate
      , ITC.itcOutputValid     = w1OutputValid
      , ITC.itcDownStreamReady = pure True
      , ITC.itcConsumeSignal   = w1ConsumeSignal
      }

  w1OutputTxn = OTC.outputTransactionController cycleCounter headIdx
    OTC.OutputTransactionIn
      { OTC.otcAllDone       = w1AllDone
      , OTC.otcConsumeSignal = w1ConsumeSignal
      }

  w1OutputValid   = OTC.otcOutputValid w1OutputTxn
  w1ConsumeSignal = w1OutputValid

  w1ReqPulse = mkRowReqPulse cycleCounter w1FetchReq w1WeightReady w1EffRow

  (w1AxiMaster, w1Lo, w1WeightValidRaw, w1WeightReadyRaw) =
    LOADER.w1WeightLoader cycleCounter w1Slave layerIdx
      w1EffRow w1ReqPulse (pure True) w1RowDone

  w1WeightValid = w1WeightValidRaw
  w1WeightReady = w1WeightReadyRaw

  w1JustConsumed = register False w1ConsumeSignal
  w1EffInput = ITC.itcLatchedValid w1InputTxn
    .&&. (not <$> w1OutputValid)
    .&&. (not <$> w1JustConsumed)

  -- Serial inline computation for W1
  (_w1MachState, w1FetchReq, _w1RowReset, _w1RowEnable, w1AllDone, _w1IdleReady,
   _w1CompCounter, w1RowDone, w1SerialResult, w1ColPrefetch) =
    serialRowAccum @dom @HiddenDimension @ModelDimension
      w1EffInput w1WeightValid w1RowDone
      w1RowIndex (LOADER.dramRowOut w1Lo) xHatBramRdData

  -- Write each completed W1 row result to FFN BRAM slot A.
  w1BramWriteOp :: Signal dom (Maybe (FFNBramAddr, FixedPoint))
  w1BramWriteOp = mux w1RowDone
    (Just <$> ((,) <$> (fromIntegral <$> w1RowIndex) <*> w1SerialResult))
    (pure Nothing)

  -------------------------------------------------------------------------
  -- W3 (Up) Phase — HiddenDimension rows × ModelDimension cols
  -------------------------------------------------------------------------
  w3RowIndex :: Signal dom (Index HiddenDimension)
  w3RowIndex = register 0 (RS.rsNextRowIndex w3RS)

  w3RS = RS.rowScheduler RS.RowSchedulerIn
    { RS.rsRowDone       = w3RowDone
    , RS.rsOutputValid   = w3OutputValid
    , RS.rsConsumeSignal = w3ConsumeSignal
    , RS.rsCurrentIndex  = w3RowIndex
    }

  w3EffRow :: Signal dom (Index HiddenDimension)
  w3EffRow = mux (w3OutputValid .&&. w3ConsumeSignal) (pure 0) w3RowIndex

  w3InputTxn = ITC.inputTransactionController cycleCounter headIdx
    ITC.InputTransactionIn
      { ITC.itcInputValid      = fpState .==. pure FPUp
      , ITC.itcOutputValid     = w3OutputValid
      , ITC.itcDownStreamReady = pure True
      , ITC.itcConsumeSignal   = w3ConsumeSignal
      }

  w3OutputTxn = OTC.outputTransactionController cycleCounter headIdx
    OTC.OutputTransactionIn
      { OTC.otcAllDone       = w3AllDone
      , OTC.otcConsumeSignal = w3ConsumeSignal
      }

  w3OutputValid   = OTC.otcOutputValid w3OutputTxn
  w3ConsumeSignal = w3OutputValid

  w3ReqPulse = mkRowReqPulse cycleCounter w3FetchReq w3WeightReady w3EffRow

  (w3AxiMaster, w3Lo, w3WeightValidRaw, w3WeightReadyRaw) =
    LOADER.w3WeightLoader cycleCounter w3Slave layerIdx
      w3EffRow w3ReqPulse (pure True) w3RowDone

  w3WeightValid = w3WeightValidRaw
  w3WeightReady = w3WeightReadyRaw

  w3JustConsumed = register False w3ConsumeSignal
  w3EffInput = ITC.itcLatchedValid w3InputTxn
    .&&. (not <$> w3OutputValid)
    .&&. (not <$> w3JustConsumed)

  -- Serial inline computation for W3
  (_w3MachState, w3FetchReq, _w3RowReset, _w3RowEnable, w3AllDone, _w3IdleReady,
   _w3CompCounter, w3RowDone, w3SerialResult, w3ColPrefetch) =
    serialRowAccum @dom @HiddenDimension @ModelDimension
      w3EffInput w3WeightValid w3RowDone
      w3RowIndex (LOADER.dramRowOut w3Lo) xHatBramRdData

  -- Latch W3 row result and index for 1-cycle FFN BRAM read latency.
  w3ResultLatch :: Signal dom FixedPoint
  w3ResultLatch = regEn 0 w3RowDone w3SerialResult

  w3RowIdxLatch :: Signal dom (Index HiddenDimension)
  w3RowIdxLatch = regEn 0 w3RowDone w3RowIndex

  -- 1-cycle delay: FFN BRAM read for slot A[i] was issued on w3RowDone;
  -- data arrives the next cycle. Compute SiLU(gate[i]) * up[i] → slot B[i].
  w3WriteEnabled :: Signal dom Bool
  w3WriteEnabled = register False w3RowDone

  w3SiluBramWriteOp :: Signal dom (Maybe (FFNBramAddr, FixedPoint))
  w3SiluBramWriteOp = mux w3WriteEnabled
    (Just <$> ((,)
        <$> ((ffnSlotBBase +) . fromIntegral <$> w3RowIdxLatch)
        <*> ((*) <$> (sigmoidLinearUnit <$> ffnBramRdData) <*> w3ResultLatch)))
    (pure Nothing)

  -------------------------------------------------------------------------
  -- W2 (Down) Phase — ModelDimension rows × HiddenDimension cols
  --
  -- Inline serial: column read from FFN BRAM slot B, results to slot C.
  -------------------------------------------------------------------------
  w2RowIndex :: Signal dom (Index ModelDimension)
  w2RowIndex = register 0 (RS.rsNextRowIndex w2RS)

  w2RS = RS.rowScheduler RS.RowSchedulerIn
    { RS.rsRowDone       = w2RowDone
    , RS.rsOutputValid   = w2OutputValid
    , RS.rsConsumeSignal = w2ConsumeSignal
    , RS.rsCurrentIndex  = w2RowIndex
    }

  w2EffRow :: Signal dom (Index ModelDimension)
  w2EffRow = mux (w2OutputValid .&&. w2ConsumeSignal) (pure 0) w2RowIndex

  w2InputTxn = ITC.inputTransactionController cycleCounter headIdx
    ITC.InputTransactionIn
      { ITC.itcInputValid      = fpState .==. pure FPDown
      , ITC.itcOutputValid     = w2OutputValid
      , ITC.itcDownStreamReady = readyIn
      , ITC.itcConsumeSignal   = w2ConsumeSignal
      }

  w2OutputTxn = OTC.outputTransactionController cycleCounter headIdx
    OTC.OutputTransactionIn
      { OTC.otcAllDone       = w2AllDone
      , OTC.otcConsumeSignal = w2ConsumeSignal
      }

  w2OutputValid   = OTC.otcOutputValid w2OutputTxn
  w2ConsumeSignal = w2OutputValid

  w2ReqPulse = mkRowReqPulse cycleCounter w2FetchReq w2WeightReady w2EffRow

  (w2AxiMaster, w2Lo, w2WeightValidRaw, w2WeightReadyRaw) =
    LOADER.w2WeightLoader cycleCounter w2Slave layerIdx
      w2EffRow w2ReqPulse (pure True) w2RowDone

  w2WeightValid = w2WeightValidRaw
  w2WeightReady = w2WeightReadyRaw

  w2JustConsumed = register False w2ConsumeSignal
  w2EffInput = ITC.itcLatchedValid w2InputTxn
    .&&. (not <$> w2OutputValid)
    .&&. (not <$> w2JustConsumed)

  (_w2MachState :: Signal dom MultiplierState, w2FetchReq, w2RowReset, w2RowEnable, w2AllDone, _w2IdleReady) =
    matrixMultiplierStateMachine
      w2EffInput w2WeightValid (pure True) w2RowDone w2RowIndex

  w2CompCounter :: Signal dom (Index HiddenDimension)
  w2CompCounter = register 0 nextW2CompCounter

  nextW2CompCounter :: Signal dom (Index HiddenDimension)
  nextW2CompCounter =
    mux w2RowReset  (pure 0) $
    mux w2RowEnable (satSucc SatBound <$> w2CompCounter) $
    w2CompCounter

  -- BRAM pre-fetch for slot B column during FPDown
  w2BramPrefetch :: Signal dom (Index HiddenDimension)
  w2BramPrefetch =
    mux w2RowReset (pure 0) (satSucc SatBound <$> w2CompCounter)

  w2WeightRow :: Signal dom (RowI8E HiddenDimension)
  w2WeightRow = LOADER.dramRowOut w2Lo

  w2MantissaElem :: Signal dom (Signed 8)
  w2MantissaElem = (!!) <$> (rowMantissas <$> w2WeightRow) <*> w2CompCounter

  w2Product :: Signal dom FixedPoint
  w2Product = (fromIntegral <$> w2MantissaElem) * ffnBramRdData

  w2Acc :: Signal dom FixedPoint
  w2Acc = register 0 nextW2Acc

  nextW2Acc :: Signal dom FixedPoint
  nextW2Acc =
    mux w2RowReset (pure 0) $
    mux (w2RowEnable .&&. (not <$> w2RowDone)) (w2Acc + w2Product) $
    w2Acc

  w2LastElemFlag :: Signal dom Bool
  w2LastElemFlag = (w2CompCounter .==. pure maxBound) .&&. w2RowEnable

  w2RowDoneRaw :: Signal dom Bool
  w2RowDoneRaw = w2LastElemFlag .&&. (not <$> register False w2LastElemFlag)

  w2RowDone :: Signal dom Bool
  w2RowDone = register False w2RowDoneRaw

  w2SerialResult :: Signal dom FixedPoint
  w2SerialResult = scalePow2F <$> (rowExponent <$> w2WeightRow) <*> w2Acc

  w2BramWriteOp :: Signal dom (Maybe (FFNBramAddr, FixedPoint))
  w2BramWriteOp = mux w2RowDone
    (Just <$> ((,) <$> ((ffnSlotCBase +) . fromIntegral <$> w2RowIndex) <*> w2SerialResult))
    (pure Nothing)

  -------------------------------------------------------------------------
  -- Top-level handshaking
  -------------------------------------------------------------------------
  validOut = fpState .==. pure FPDone
  readyOut = (fpState .==. pure FPIdle) .&&. w1WeightReady
