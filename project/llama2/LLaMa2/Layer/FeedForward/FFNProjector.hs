module LLaMa2.Layer.FeedForward.FFNProjector
  ( ffnProjector
  ) where

import Clash.Prelude
import qualified Prelude as P

import LLaMa2.Types.ModelConfig
    ( ModelDimension, HiddenDimension, NumLayers, NumQueryHeads )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layer.FeedForward.Activation (sigmoidLinearUnit)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Memory.AXI.Slave  as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Arbiter as ARB

import qualified LLaMa2.Layer.Attention.WeightLoader as LOADER
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OTC
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator           as OA
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController  as ITC
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit              as RCU
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler                as RS

import TraceUtils (traceEdgeC)

--------------------------------------------------------------------------------
-- FFN Phase FSM
--------------------------------------------------------------------------------

data FFNProjState = FPIdle | FPGate | FPUp | FPDown | FPDone
  deriving (Show, Eq, Generic, NFDataX)

--------------------------------------------------------------------------------
-- Row Result Checker
-- Validates DRAM row dot-product matches HC row dot-product on each rowDone.
--------------------------------------------------------------------------------

rowResultChecker :: forall dom numRows.
  ( HiddenClockResetEnable dom, KnownNat numRows )
  => String
  -> Signal dom Bool
  -> Signal dom (Index numRows)
  -> Signal dom FixedPoint
  -> Signal dom FixedPoint
  -> Signal dom FixedPoint
rowResultChecker tag rowDone rowIdx dramResult hcResult = result
  where
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt = register 0
      (mux (rowDone .&&. (rowIdx .==. pure maxBound)) (tokenCnt + 1) tokenCnt)

    result = mux rowDone
               (check <$> tokenCnt <*> rowIdx <*> dramResult <*> hcResult)
               dramResult

    check tok ri dr hr
      | dr P.== hr = dr
      | otherwise  = P.error $ tag P.++ " row mismatch at token " P.++ show tok
                     P.++ " row " P.++ show ri
                     P.++ ": DRAM=" P.++ show dr P.++ " HC=" P.++ show hr

--------------------------------------------------------------------------------
-- Output Checker (rising-edge capture, avoids stale-data on one-cycle pulse)
--------------------------------------------------------------------------------

ffnOutputChecker :: forall dom n.
  ( HiddenClockResetEnable dom, KnownNat n )
  => String
  -> Signal dom Bool
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
  -> Signal dom (Vec n FixedPoint)
ffnOutputChecker tag outputValid dramOut hcOut = result
  where
    risingEdge   = outputValid .&&. (not <$> register False outputValid)
    dramSampled  = register (repeat 0) (mux risingEdge dramOut  dramSampled)
    hcSampled    = register (repeat 0) (mux risingEdge hcOut    hcSampled)
    tokenCnt :: Signal dom (Unsigned 32)
    tokenCnt     = register 0 (mux risingEdge (tokenCnt + 1) tokenCnt)
    checkTrigger = register False risingEdge
    result       = mux checkTrigger
                     (checkPure <$> tokenCnt <*> dramSampled <*> hcSampled)
                     dramOut

    checkPure tok dr hr =
      let pairs      = P.zip [0..] (P.zip (toList dr) (toList hr))
          mismatches = P.filter (\(_, (d, h)) -> d P./= h) pairs
      in if P.null mismatches then dr
         else let (i, (d, h)) = P.head mismatches
              in P.error $ tag P.++ " output mismatch at token " P.++ show tok
                        P.++ ": idx " P.++ show (i :: Int)
                        P.++ " (DRAM=" P.++ show d P.++ ", HC=" P.++ show h P.++ ")"

--------------------------------------------------------------------------------
-- Row request pulse helper (same pattern as WOHeadProjector)
--------------------------------------------------------------------------------

mkRowReqPulse :: forall dom numRows.
  ( HiddenClockResetEnable dom, KnownNat numRows )
  => String
  -> Signal dom (Unsigned 32)
  -> Signal dom Bool              -- ^ rcFetchReq from RowComputeUnit
  -> Signal dom Bool              -- ^ weightReady
  -> Signal dom (Index numRows)   -- ^ effectiveRowIndex
  -> Signal dom Bool
mkRowReqPulse tag cycleCounter fetchReq weightReady effRowIdx = pulse
  where
    loaderBecameIdle = weightReady .&&. (not <$> register False weightReady)
    reqValidGated    = fetchReq .&&. weightReady
    prevReqValid     = register False $ mux loaderBecameIdle (pure False) reqValidGated
    reqRise          = reqValidGated .&&. (not <$> prevReqValid)
    prevRowIdx       = register (0 :: Index numRows) effRowIdx
    rowIdxChanged    = effRowIdx ./=. prevRowIdx
    pulse            = traceEdgeC cycleCounter tag
                         (reqRise .||. (reqValidGated .&&. rowIdxChanged))

--------------------------------------------------------------------------------
-- ffnProjector
--
-- DRAM-backed replacement for feedForwardCore. Sequential phases:
--   FPGate: W1 (gate)   — HiddenDimension × ModelDimension, column = xHat
--   FPUp:   W3 (up)     — HiddenDimension × ModelDimension, column = xHat
--   FPDown: W2 (down)   — ModelDimension  × HiddenDimension, column = SiLU(W1) ⊙ W3
--
-- Each phase self-consumes immediately on completion. The final W2 result is
-- held in 'outputResult' while the FSM stays in FPDone until readyIn.
--------------------------------------------------------------------------------

ffnProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Index NumLayers
  -> Signal dom Bool                              -- ^ validIn
  -> Signal dom Bool                              -- ^ readyIn (from downstream)
  -> Signal dom (Vec ModelDimension FixedPoint)   -- ^ xHat (RMS-normalised input)
  -> PARAM.DecoderParameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec ModelDimension FixedPoint) -- ^ W2 output (before residual)
     , Signal dom Bool                            -- ^ validOut
     , Signal dom Bool                            -- ^ readyOut
     )
ffnProjector cycleCounter dramSlaveIn layerIdx validIn readyIn xHat params =
  (axiMasterOut, outputResult, validOut, readyOut)
 where
  tag     = "[FFN L" P.++ show layerIdx P.++ "] "
  headIdx = 0 :: Index NumQueryHeads

  -------------------------------------------------------------------------
  -- 3-master AXI sub-arbiter: slot 0 = W1, slot 1 = W3, slot 2 = W2
  -------------------------------------------------------------------------
  (axiMasterOut, perPhaseSlaves) =
    ARB.axiArbiterWithRouting cycleCounter dramSlaveIn
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

  -- Latch xHat when we accept a new token
  xHatLatched :: Signal dom (Vec ModelDimension FixedPoint)
  xHatLatched = regEn (repeat 0) acceptInput xHat

  -------------------------------------------------------------------------
  -- W1 (Gate) Phase — HiddenDimension rows × ModelDimension cols
  -------------------------------------------------------------------------
  w1RowIndex :: Signal dom (Index HiddenDimension)
  w1RowIndex = register 0 (RS.rsNextRowIndex w1RS)

  w1RS = RS.rowScheduler RS.RowSchedulerIn
    { RS.rsRowDone       = RCU.rcRowDone w1Compute
    , RS.rsOutputValid   = w1OutputValid
    , RS.rsConsumeSignal = w1ConsumeSignal
    , RS.rsCurrentIndex  = w1RowIndex
    }

  -- Reset row index combinatorially on consume
  w1EffRow :: Signal dom (Index HiddenDimension)
  w1EffRow = mux (w1OutputValid .&&. w1ConsumeSignal) (pure 0) w1RowIndex

  w1InputTxn = ITC.inputTransactionController cycleCounter layerIdx headIdx
    ITC.InputTransactionIn
      { ITC.itcInputValid      = fpState .==. pure FPGate
      , ITC.itcOutputValid     = w1OutputValid
      , ITC.itcDownStreamReady = pure True
      , ITC.itcConsumeSignal   = w1ConsumeSignal
      }

  w1OutputTxn = OTC.outputTransactionController cycleCounter layerIdx headIdx
    OTC.OutputTransactionIn
      { OTC.otcAllDone       = RCU.rcAllDone w1Compute
      , OTC.otcConsumeSignal = w1ConsumeSignal
      }

  w1OutputValid   = OTC.otcOutputValid w1OutputTxn
  w1ConsumeSignal = w1OutputValid  -- immediate self-consume (one-cycle pulse)

  w1ReqPulse = mkRowReqPulse (tag P.++ "w1Req") cycleCounter
                 (RCU.rcFetchReq w1Compute) w1WeightReady w1EffRow

  (w1AxiMaster, w1Lo, w1WeightValidRaw, w1WeightReadyRaw) =
    LOADER.w1WeightLoader cycleCounter w1Slave layerIdx
      w1EffRow w1ReqPulse (pure True) (RCU.rcRowDone w1Compute) params

  w1WeightValid = traceEdgeC cycleCounter (tag P.++ "w1wV") w1WeightValidRaw
  w1WeightReady = traceEdgeC cycleCounter (tag P.++ "w1wR") w1WeightReadyRaw

  w1JustConsumed = register False w1ConsumeSignal
  w1EffInput = ITC.itcLatchedValid w1InputTxn
    .&&. (not <$> w1OutputValid)
    .&&. (not <$> w1JustConsumed)

  w1Compute = RCU.rowComputeUnit cycleCounter RCU.RowComputeIn
    { RCU.rcInputValid      = w1EffInput
    , RCU.rcWeightValid     = w1WeightValid
    , RCU.rcDownStreamReady = pure True
    , RCU.rcRowIndex        = w1RowIndex
    , RCU.rcWeightDram      = LOADER.assertRowStable w1WeightValid (LOADER.dramRowOut w1Lo)
    , RCU.rcWeightHC        = LOADER.assertRowStable w1WeightValid (LOADER.hcRowOut   w1Lo)
    , RCU.rcColumn          = xHatLatched
    }

  w1DramChecked = rowResultChecker (tag P.++ "W1")
    (RCU.rcRowDone w1Compute) w1RowIndex
    (RCU.rcResult   w1Compute)
    (RCU.rcResultHC w1Compute)

  w1Accum = OA.outputAccumulator cycleCounter layerIdx headIdx OA.OutputAccumIn
    { OA.oaRowDone     = RCU.rcRowDone w1Compute
    , OA.oaRowIndex    = w1RowIndex
    , OA.oaRowResult   = w1DramChecked
    , OA.oaRowResultHC = RCU.rcResultHC w1Compute
    }

  -- Latch W1 output; apply SiLU
  gateRaw :: Signal dom (Vec HiddenDimension FixedPoint)
  gateRaw = regEn (repeat 0) w1OutputValid (OA.oaOutput w1Accum)

  gateSiLU :: Signal dom (Vec HiddenDimension FixedPoint)
  gateSiLU = map sigmoidLinearUnit <$> gateRaw

  -------------------------------------------------------------------------
  -- W3 (Up) Phase — HiddenDimension rows × ModelDimension cols
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

  w3InputTxn = ITC.inputTransactionController cycleCounter layerIdx headIdx
    ITC.InputTransactionIn
      { ITC.itcInputValid      = fpState .==. pure FPUp
      , ITC.itcOutputValid     = w3OutputValid
      , ITC.itcDownStreamReady = pure True
      , ITC.itcConsumeSignal   = w3ConsumeSignal
      }

  w3OutputTxn = OTC.outputTransactionController cycleCounter layerIdx headIdx
    OTC.OutputTransactionIn
      { OTC.otcAllDone       = RCU.rcAllDone w3Compute
      , OTC.otcConsumeSignal = w3ConsumeSignal
      }

  w3OutputValid   = OTC.otcOutputValid w3OutputTxn
  w3ConsumeSignal = w3OutputValid

  w3ReqPulse = mkRowReqPulse (tag P.++ "w3Req") cycleCounter
                 (RCU.rcFetchReq w3Compute) w3WeightReady w3EffRow

  (w3AxiMaster, w3Lo, w3WeightValidRaw, w3WeightReadyRaw) =
    LOADER.w3WeightLoader cycleCounter w3Slave layerIdx
      w3EffRow w3ReqPulse (pure True) (RCU.rcRowDone w3Compute) params

  w3WeightValid = traceEdgeC cycleCounter (tag P.++ "w3wV") w3WeightValidRaw
  w3WeightReady = traceEdgeC cycleCounter (tag P.++ "w3wR") w3WeightReadyRaw

  w3JustConsumed = register False w3ConsumeSignal
  w3EffInput = ITC.itcLatchedValid w3InputTxn
    .&&. (not <$> w3OutputValid)
    .&&. (not <$> w3JustConsumed)

  w3Compute = RCU.rowComputeUnit cycleCounter RCU.RowComputeIn
    { RCU.rcInputValid      = w3EffInput
    , RCU.rcWeightValid     = w3WeightValid
    , RCU.rcDownStreamReady = pure True
    , RCU.rcRowIndex        = w3RowIndex
    , RCU.rcWeightDram      = LOADER.assertRowStable w3WeightValid (LOADER.dramRowOut w3Lo)
    , RCU.rcWeightHC        = LOADER.assertRowStable w3WeightValid (LOADER.hcRowOut   w3Lo)
    , RCU.rcColumn          = xHatLatched
    }

  w3DramChecked = rowResultChecker (tag P.++ "W3")
    (RCU.rcRowDone w3Compute) w3RowIndex
    (RCU.rcResult   w3Compute)
    (RCU.rcResultHC w3Compute)

  w3Accum = OA.outputAccumulator cycleCounter layerIdx headIdx OA.OutputAccumIn
    { OA.oaRowDone     = RCU.rcRowDone w3Compute
    , OA.oaRowIndex    = w3RowIndex
    , OA.oaRowResult   = w3DramChecked
    , OA.oaRowResultHC = RCU.rcResultHC w3Compute
    }

  -- Compute gate ⊙ up at the w3OutputValid cycle.
  -- At that cycle oaOutput w3Accum holds the complete W3 result,
  -- and gateSiLU holds the already-latched SiLU-activated W1 output.
  gateUpLatched :: Signal dom (Vec HiddenDimension FixedPoint)
  gateUpLatched = regEn (repeat 0) w3OutputValid
    (zipWith (*) <$> gateSiLU <*> OA.oaOutput w3Accum)

  -------------------------------------------------------------------------
  -- W2 (Down) Phase — ModelDimension rows × HiddenDimension cols
  -------------------------------------------------------------------------
  w2RowIndex :: Signal dom (Index ModelDimension)
  w2RowIndex = register 0 (RS.rsNextRowIndex w2RS)

  w2RS = RS.rowScheduler RS.RowSchedulerIn
    { RS.rsRowDone       = RCU.rcRowDone w2Compute
    , RS.rsOutputValid   = w2OutputValid
    , RS.rsConsumeSignal = w2ConsumeSignal
    , RS.rsCurrentIndex  = w2RowIndex
    }

  w2EffRow :: Signal dom (Index ModelDimension)
  w2EffRow = mux (w2OutputValid .&&. w2ConsumeSignal) (pure 0) w2RowIndex

  w2InputTxn = ITC.inputTransactionController cycleCounter layerIdx headIdx
    ITC.InputTransactionIn
      { ITC.itcInputValid      = fpState .==. pure FPDown
      , ITC.itcOutputValid     = w2OutputValid
      , ITC.itcDownStreamReady = readyIn
      , ITC.itcConsumeSignal   = w2ConsumeSignal
      }

  w2OutputTxn = OTC.outputTransactionController cycleCounter layerIdx headIdx
    OTC.OutputTransactionIn
      { OTC.otcAllDone       = RCU.rcAllDone w2Compute
      , OTC.otcConsumeSignal = w2ConsumeSignal
      }

  -- Immediate self-consume: result is held in outputResult (registered below)
  w2OutputValid   = OTC.otcOutputValid w2OutputTxn
  w2ConsumeSignal = w2OutputValid

  w2ReqPulse = mkRowReqPulse (tag P.++ "w2Req") cycleCounter
                 (RCU.rcFetchReq w2Compute) w2WeightReady w2EffRow

  (w2AxiMaster, w2Lo, w2WeightValidRaw, w2WeightReadyRaw) =
    LOADER.w2WeightLoader cycleCounter w2Slave layerIdx
      w2EffRow w2ReqPulse (pure True) (RCU.rcRowDone w2Compute) params

  w2WeightValid = traceEdgeC cycleCounter (tag P.++ "w2wV") w2WeightValidRaw
  w2WeightReady = traceEdgeC cycleCounter (tag P.++ "w2wR") w2WeightReadyRaw

  w2JustConsumed = register False w2ConsumeSignal
  w2EffInput = ITC.itcLatchedValid w2InputTxn
    .&&. (not <$> w2OutputValid)
    .&&. (not <$> w2JustConsumed)

  -- Column for W2 is gateUpLatched (SiLU(W1) ⊙ W3).
  -- Both DRAM and HC paths use this same column; the checker validates W2 weights.
  w2Compute = RCU.rowComputeUnit cycleCounter RCU.RowComputeIn
    { RCU.rcInputValid      = w2EffInput
    , RCU.rcWeightValid     = w2WeightValid
    , RCU.rcDownStreamReady = readyIn
    , RCU.rcRowIndex        = w2RowIndex
    , RCU.rcWeightDram      = LOADER.assertRowStable w2WeightValid (LOADER.dramRowOut w2Lo)
    , RCU.rcWeightHC        = LOADER.assertRowStable w2WeightValid (LOADER.hcRowOut   w2Lo)
    , RCU.rcColumn          = gateUpLatched
    }

  w2DramChecked = rowResultChecker (tag P.++ "W2")
    (RCU.rcRowDone w2Compute) w2RowIndex
    (RCU.rcResult   w2Compute)
    (RCU.rcResultHC w2Compute)

  w2Accum = OA.outputAccumulator cycleCounter layerIdx headIdx OA.OutputAccumIn
    { OA.oaRowDone     = RCU.rcRowDone w2Compute
    , OA.oaRowIndex    = w2RowIndex
    , OA.oaRowResult   = w2DramChecked
    , OA.oaRowResultHC = RCU.rcResultHC w2Compute
    }

  w2OutFinal = ffnOutputChecker (tag P.++ "W2out") w2OutputValid
                 (OA.oaOutput   w2Accum)
                 (OA.oaOutputHC w2Accum)

  -- Latch W2 output; held for the duration of FPDone state
  outputResult :: Signal dom (Vec ModelDimension FixedPoint)
  outputResult = regEn (repeat 0) w2OutputValid w2OutFinal

  -------------------------------------------------------------------------
  -- Top-level handshaking
  -------------------------------------------------------------------------
  validOut = fpState .==. pure FPDone
  readyOut = (fpState .==. pure FPIdle) .&&. w1WeightReady
