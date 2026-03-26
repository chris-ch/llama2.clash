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
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit              as RCU
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler                as RS

--------------------------------------------------------------------------------
-- FFN intermediate BRAM address types
--
-- Slot A [0 .. HiddenDim-1]           : w1 (gate) results
-- Slot B [HiddenDim .. 2*HiddenDim-1] : SiLU(gate)*up product (w2 column input)
--
-- Using a separate BRAM eliminates:
--   • w1Accum Vec HiddenDimension register (gateRaw)
--   • w3Accum Vec HiddenDimension register (gateUpLatched)
--   • gateRaw   = regEn (repeat 0) w1OutputValid (oaOutput w1Accum)
--   • gateUpLatched = regEn ... w3OutputValid (zipWith (*) gateSiLU ...)
--------------------------------------------------------------------------------

-- Slot A [0 .. HiddenDim-1]              : w1 (gate) results
-- Slot B [HiddenDim .. 2*HiddenDim-1]    : SiLU(gate)*up product
-- Slot C [2*HiddenDim .. 2*HiddenDim+ModelDim-1] : w2 (down) results
type FFNBramDepth = 2 TN.* HiddenDimension TN.+ ModelDimension
type FFNBramAddr  = Index FFNBramDepth

-- Slot A base address is 0 (implicit in address arithmetic below).
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
-- ffnProjector
--
-- DRAM-backed FFN with BRAM-backed intermediate storage.
-- Sequential phases:
--   FPGate: W1 (gate)   — HiddenDimension × ModelDimension, column = xHat
--           Results written element-by-element to FFN BRAM slot A.
--   FPUp:   W3 (up)     — HiddenDimension × ModelDimension, column = xHat
--           As each row i completes, reads slot A[i] from BRAM, computes
--           SiLU(gate[i]) * up[i], writes to slot B[i].
--   FPDown: W2 (down)   — ModelDimension × HiddenDimension
--           Column (slot B) is read serially from BRAM, one element per cycle.
--           No Vec HiddenDimension register is held across phases.
--------------------------------------------------------------------------------

ffnProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Signal dom Bool                                          -- ^ validIn
  -> Signal dom Bool                                          -- ^ readyIn (from downstream, gated by caller until residual done)
  -> Signal dom (Maybe (Index ModelDimension, FixedPoint))    -- ^ xHatWrite: rmsNorm BRAM writes (element-by-element)
  -> Signal dom (Index ModelDimension)                        -- ^ ffnCRdAddr: slot C read address driven by caller during FPDone
  -> ( Master.AxiMasterOut dom
     , Signal dom FixedPoint                      -- ^ FFN BRAM read data (slot C, for caller's residual FSM)
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
  -- xHat BRAM: written element-by-element by rmsNorm (before acceptInput).
  -- Read port time-multiplexed: w1 during FPGate, w3 during FPUp.
  -------------------------------------------------------------------------
  xHatRdAddr :: Signal dom (Index ModelDimension)
  xHatRdAddr = mux (fpState .==. pure FPGate)
    (RCU.rcColumnAddr w1Compute)
    (RCU.rcColumnAddr w3Compute)

  xHatRdData :: Signal dom FixedPoint
  xHatRdData = blockRam (repeat 0 :: Vec ModelDimension FixedPoint)
                 xHatRdAddr xHatWrite

  -------------------------------------------------------------------------
  -- FFN intermediate BRAM
  --   Port A: read  (gate reads during FPUp, column reads during FPDown)
  --   Port B: write (w1 results during FPGate, SiLU*up during FPUp)
  -------------------------------------------------------------------------
  ffnBramRdData :: Signal dom FixedPoint
  ffnBramRdData = fst $ trueDualPortRam
    ffnBramRdAddr
    (pure Nothing)          -- port A: read-only
    ffnBramWrAddr
    ffnBramWriteOp          -- port B: write when Just

  -- Write mux: phases are strictly sequential so these are never simultaneous.
  --   w1 writes (FPGate): slot A[rowIndex] ← gate result
  --   w3 SiLU writes (1 cycle after FPUp row done): slot B[rowIndex] ← SiLU*up
  --   w2 writes (FPDown, each row done): slot C[rowIndex] ← w2 serial result
  ffnBramWriteOp :: Signal dom (Maybe (FFNBramAddr, FixedPoint))
  ffnBramWriteOp =
    mux (RCU.rcRowDone w1Compute) w1BramWriteOp $
    mux w3WriteEnabled w3SiluBramWriteOp $
    w2BramWriteOp

  ffnBramWrAddr :: Signal dom FFNBramAddr
  ffnBramWrAddr = maybe 0 fst <$> ffnBramWriteOp

  -- Read address mux:
  --   FPDown  → slot B + serial column counter (1-cycle pre-fetch)
  --   FPDone  → slot C + ffnCRdAddr  (caller reads w2 results for residual)
  --   Other   → slot A + w3RowIndex  (FPGate/FPUp/FPIdle, harmless default)
  ffnBramRdAddr :: Signal dom FFNBramAddr
  ffnBramRdAddr =
    mux (fpState .==. pure FPDown)
      ((ffnSlotBBase +) . fromIntegral <$> w2BramPrefetch)
    $ mux (fpState .==. pure FPDone)
      ((ffnSlotCBase +) . fromIntegral <$> ffnCRdAddr)
    $ fromIntegral <$> w3RowIndex

  -------------------------------------------------------------------------
  -- W1 (Gate) Phase — HiddenDimension rows × ModelDimension cols
  -- Row results written to FFN BRAM slot A instead of OutputAccumulator.
  -------------------------------------------------------------------------
  w1RowIndex :: Signal dom (Index HiddenDimension)
  w1RowIndex = register 0 (RS.rsNextRowIndex w1RS)

  w1RS = RS.rowScheduler RS.RowSchedulerIn
    { RS.rsRowDone       = RCU.rcRowDone w1Compute
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
      { OTC.otcAllDone       = RCU.rcAllDone w1Compute
      , OTC.otcConsumeSignal = w1ConsumeSignal
      }

  w1OutputValid   = OTC.otcOutputValid w1OutputTxn
  w1ConsumeSignal = w1OutputValid

  w1ReqPulse = mkRowReqPulse cycleCounter
                 (RCU.rcFetchReq w1Compute) w1WeightReady w1EffRow

  (w1AxiMaster, w1Lo, w1WeightValidRaw, w1WeightReadyRaw) =
    LOADER.w1WeightLoader cycleCounter w1Slave layerIdx
      w1EffRow w1ReqPulse (pure True) (RCU.rcRowDone w1Compute)

  w1WeightValid = w1WeightValidRaw
  w1WeightReady = w1WeightReadyRaw

  w1JustConsumed = register False w1ConsumeSignal
  w1EffInput = ITC.itcLatchedValid w1InputTxn
    .&&. (not <$> w1OutputValid)
    .&&. (not <$> w1JustConsumed)

  w1Compute = RCU.rowComputeUnit cycleCounter RCU.RowComputeIn
    { RCU.rcInputValid      = w1EffInput
    , RCU.rcWeightValid     = w1WeightValid
    , RCU.rcDownStreamReady = pure True
    , RCU.rcRowIndex        = w1RowIndex
    , RCU.rcWeightDram      = LOADER.dramRowOut w1Lo
    , RCU.rcColumnRdData    = xHatRdData
    }

  -- Element-by-element write to slot A: fires once per row on rcRowDone.
  w1BramWriteOp :: Signal dom (Maybe (FFNBramAddr, FixedPoint))
  w1BramWriteOp = mux (RCU.rcRowDone w1Compute)
    (Just <$> ((,) <$> (fromIntegral <$> w1RowIndex) <*> RCU.rcResult w1Compute))
    (pure Nothing)

  -------------------------------------------------------------------------
  -- W3 (Up) Phase — HiddenDimension rows × ModelDimension cols
  -- As each row i completes, reads gate[i] from slot A (1-cycle BRAM
  -- latency), computes SiLU(gate[i]) * up[i], writes to slot B[i].
  -------------------------------------------------------------------------
  w3RowIndex :: Signal dom (Index HiddenDimension)
  w3RowIndex = register 0 (RS.rsNextRowIndex w3RS)

  w3RS = RS.rowScheduler RS.RowSchedulerIn
    { RS.rsRowDone       = RCU.rcRowDone w3Compute
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
      { OTC.otcAllDone       = RCU.rcAllDone w3Compute
      , OTC.otcConsumeSignal = w3ConsumeSignal
      }

  w3OutputValid   = OTC.otcOutputValid w3OutputTxn
  w3ConsumeSignal = w3OutputValid

  w3ReqPulse = mkRowReqPulse cycleCounter
                 (RCU.rcFetchReq w3Compute) w3WeightReady w3EffRow

  (w3AxiMaster, w3Lo, w3WeightValidRaw, w3WeightReadyRaw) =
    LOADER.w3WeightLoader cycleCounter w3Slave layerIdx
      w3EffRow w3ReqPulse (pure True) (RCU.rcRowDone w3Compute)

  w3WeightValid = w3WeightValidRaw
  w3WeightReady = w3WeightReadyRaw

  w3JustConsumed = register False w3ConsumeSignal
  w3EffInput = ITC.itcLatchedValid w3InputTxn
    .&&. (not <$> w3OutputValid)
    .&&. (not <$> w3JustConsumed)

  w3Compute = RCU.rowComputeUnit cycleCounter RCU.RowComputeIn
    { RCU.rcInputValid      = w3EffInput
    , RCU.rcWeightValid     = w3WeightValid
    , RCU.rcDownStreamReady = pure True
    , RCU.rcRowIndex        = w3RowIndex
    , RCU.rcWeightDram      = LOADER.dramRowOut w3Lo
    , RCU.rcColumnRdData    = xHatRdData
    }

  -- Latch w3 row result and index for 1-cycle BRAM read latency.
  w3RowDone :: Signal dom Bool
  w3RowDone = RCU.rcRowDone w3Compute

  w3ResultLatch :: Signal dom FixedPoint
  w3ResultLatch = regEn 0 w3RowDone (RCU.rcResult w3Compute)

  w3RowIdxLatch :: Signal dom (Index HiddenDimension)
  w3RowIdxLatch = regEn 0 w3RowDone w3RowIndex

  -- 1-cycle delay: BRAM read for slot A[i] was issued when w3RowDone fired;
  -- data arrives this cycle. Compute and write slot B[i].
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
  -- The column (formerly gateUpLatched :: Vec HiddenDimension FixedPoint)
  -- is now read serially from FFN BRAM slot B, one element per cycle.
  -- This replaces RCU.rowComputeUnit with an inline serial dot product.
  --
  -- Per-row timing:
  --   MReset   (1 cycle): w2CompCounter reset, BRAM pre-fetches col[0]
  --   MProcess (HiddenDimension cycles):
  --     cycle k (k=0..HiddenDim-1):
  --       BRAM data = col[k] (issued at cycle k-1 / MReset)
  --       mantissa  = rowMantissas[k]
  --       acc      += mantissa[k] * col[k]
  --       BRAM pre-fetch: col[k+1] (driven via w2BramPrefetch)
  --   rowDone fires 2 cycles after last element (edge-detect + register)
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

  w2ReqPulse = mkRowReqPulse cycleCounter
                 w2FetchReq w2WeightReady w2EffRow

  (w2AxiMaster, w2Lo, w2WeightValidRaw, w2WeightReadyRaw) =
    LOADER.w2WeightLoader cycleCounter w2Slave layerIdx
      w2EffRow w2ReqPulse (pure True) w2RowDone

  w2WeightValid = w2WeightValidRaw
  w2WeightReady = w2WeightReadyRaw

  w2JustConsumed = register False w2ConsumeSignal
  w2EffInput = ITC.itcLatchedValid w2InputTxn
    .&&. (not <$> w2OutputValid)
    .&&. (not <$> w2JustConsumed)

  -- Row-level FSM (replaces matrixMultiplierStateMachine inside RCU).
  -- downStreamReady = pure True: MDone lasts exactly 1 cycle before resetting
  -- to MIdle; the OTC latches w2AllDone for us.
  (_w2MachState :: Signal dom MultiplierState, w2FetchReq, w2RowReset, w2RowEnable, w2AllDone, _w2IdleReady) =
    matrixMultiplierStateMachine
      w2EffInput w2WeightValid (pure True) w2RowDone w2RowIndex

  -- Column counter: indexes the current computation element.
  --   MReset cycle    : counter resets to 0 (applied next cycle = first MProcessing)
  --   MProcessing k   : counter = k (0-based); advances each enable cycle
  w2CompCounter :: Signal dom (Index HiddenDimension)
  w2CompCounter = register 0 nextW2CompCounter

  nextW2CompCounter :: Signal dom (Index HiddenDimension)
  nextW2CompCounter =
    mux w2RowReset (pure 0) $
    mux w2RowEnable (satSucc SatBound <$> w2CompCounter) $
    w2CompCounter

  -- BRAM pre-fetch address: drives the FFN BRAM read port during FPDown.
  -- During MReset  : 0        → col[0] arrives at first MProcessing cycle
  -- During MProcessing cycle k (compCounter=k):
  --   satSucc(k) → col[k+1] arrives next cycle (irrelevant on last element)
  w2BramPrefetch :: Signal dom (Index HiddenDimension)
  w2BramPrefetch =
    mux w2RowReset (pure 0) (satSucc SatBound <$> w2CompCounter)

  -- Weight row from DRAM (stable throughout each row's processing window).
  w2WeightRow :: Signal dom (RowI8E HiddenDimension)
  w2WeightRow = LOADER.dramRowOut w2Lo

  -- Serial multiply: mantissa[compCounter] × col[compCounter]
  -- ffnBramRdData delivers col[compCounter] with 1-cycle latency:
  --   the address issued at MReset/previous cycle was compCounter-1 (or 0),
  --   so the data arriving now is exactly col[compCounter] once aligned.
  w2MantissaElem :: Signal dom (Signed 8)
  w2MantissaElem = (!!) <$> (rowMantissas <$> w2WeightRow) <*> w2CompCounter

  w2Product :: Signal dom FixedPoint
  w2Product = (fromIntegral <$> w2MantissaElem) * ffnBramRdData

  -- Accumulator: reset on MReset, accumulate on MProcessing (guarded by rowDone).
  w2Acc :: Signal dom FixedPoint
  w2Acc = register 0 nextW2Acc

  nextW2Acc :: Signal dom FixedPoint
  nextW2Acc =
    mux w2RowReset (pure 0) $
    mux (w2RowEnable .&&. (not <$> w2RowDone)) (w2Acc + w2Product) $
    w2Acc

  -- Row done: fires when the last element (compCounter == maxBound) has been
  -- processed. Uses rising-edge detection to produce a 1-cycle pulse,
  -- mirroring parallel64RowProcessor's rowDone convention.
  w2LastElemFlag :: Signal dom Bool
  w2LastElemFlag = (w2CompCounter .==. pure maxBound) .&&. w2RowEnable

  w2RowDoneRaw :: Signal dom Bool
  w2RowDoneRaw = w2LastElemFlag .&&. (not <$> register False w2LastElemFlag)

  w2RowDone :: Signal dom Bool
  w2RowDone = register False w2RowDoneRaw

  -- Scale accumulator by quantisation exponent to produce the dot-product result.
  w2SerialResult :: Signal dom FixedPoint
  w2SerialResult = scalePow2F <$> (rowExponent <$> w2WeightRow) <*> w2Acc

  -- Write each completed row result to FFN BRAM slot C.
  -- Slot C base = 2*HiddenDimension; indexed by w2RowIndex (ModelDimension rows).
  w2BramWriteOp :: Signal dom (Maybe (FFNBramAddr, FixedPoint))
  w2BramWriteOp = mux w2RowDone
    (Just <$> ((,) <$> ((ffnSlotCBase +) . fromIntegral <$> w2RowIndex) <*> w2SerialResult))
    (pure Nothing)

  -------------------------------------------------------------------------
  -- Top-level handshaking
  -------------------------------------------------------------------------
  validOut = fpState .==. pure FPDone
  readyOut = (fpState .==. pure FPIdle) .&&. w1WeightReady
