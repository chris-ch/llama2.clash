module LLaMa2.Layer.Attention.QueryHeadProjector
  ( queryHeadProjector
  , QHeadDebugInfo(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent)
import LLaMa2.Numeric.Quantization (RowI8E (..))

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.InputTransactionController as InputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowComputeUnit as RowComputeUnit
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.RowScheduler as RowScheduler
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.WeightFetchUnit as WeightFetchUnit


--------------------------------------------------------------------------------
-- Debug Info Record
--------------------------------------------------------------------------------
data QHeadDebugInfo dom = QHeadDebugInfo
  { qhRowIndex        :: Signal dom (Index HeadDimension)
  , qhState           :: Signal dom OPS.MultiplierState
  , qhFirstMant       :: Signal dom Mantissa
  , qhRowResult       :: Signal dom FixedPoint
  , qhRowDone         :: Signal dom Bool
  , qhFetchValid      :: Signal dom Bool
  , qhFetchedWord     :: Signal dom (BitVector 512)
  , qhRowReset        :: Signal dom Bool
  , qhRowEnable       :: Signal dom Bool
  , qhAccumValue      :: Signal dom FixedPoint
  , qhQOut            :: Signal dom (Vec HeadDimension FixedPoint)
  , qhCurrentRowExp   :: Signal dom Exponent
  , qhCurrentRowMant0 :: Signal dom Mantissa
  , qhRowReqValid     :: Signal dom Bool
  , qhWeightReady     :: Signal dom Bool
  , qhWeightValid     :: Signal dom Bool
  } deriving (Generic)

--------------------------------------------------------------------------------
-- QueryHeadCore
-- Outputs a BRAM write port instead of an accumulated Vec.
-- Applies rotary encoding pair-wise as rows complete.
--------------------------------------------------------------------------------
data QueryHeadCoreOut dom = QueryHeadCoreOut
  { qhcAxiMaster   :: Master.AxiMasterOut dom
  , qhcBramWrite   :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
  , qhcOutputValid :: Signal dom Bool
  , qhcReady       :: Signal dom Bool
  , qhcDebug       :: QHeadDebugInfo dom
  }

queryHeadCore :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumQueryHeads
  -> Signal dom Bool                              -- inputValid
  -> Signal dom Bool                              -- downStreamReady
  -> Signal dom Bool                              -- consumeSignal
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint)  -- cosVec
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint)  -- sinVec
  -> Signal dom (Maybe (Index ModelDimension, FixedPoint))           -- xNormWrite
  -> QueryHeadCoreOut dom
queryHeadCore cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal cosVec sinVec xNormWrite =
  QueryHeadCoreOut
    { qhcAxiMaster   = WeightFetchUnit.wfAxiMaster weightFetch
    , qhcBramWrite   = qBramWrite
    , qhcOutputValid = OutputTransactionController.otcOutputValid outputTxn
    , qhcReady       = readyForInput
    , qhcDebug       = debugInfo
    }
  where
    rowIndex :: Signal dom (Index HeadDimension)
    rowIndex = register 0 nextRowIndex

    rsIn = RowScheduler.RowSchedulerIn
             { rsRowDone       = rowDone
             , rsOutputValid   = OutputTransactionController.otcOutputValid outputTxn
             , rsConsumeSignal = consumeSignal
             , rsCurrentIndex  = rowIndex
             }

    rowSched     = RowScheduler.rowScheduler rsIn
    nextRowIndex = RowScheduler.rsNextRowIndex rowSched

    inputTxn = InputTransactionController.inputTransactionController cycleCounter headIdx
                 InputTransactionController.InputTransactionIn
                   { itcInputValid      = inputValid
                   , itcOutputValid     = OutputTransactionController.otcOutputValid outputTxn
                   , itcDownStreamReady = downStreamReady
                   , itcConsumeSignal   = consumeSignal
                   }

    inputValidLatched = InputTransactionController.itcLatchedValid inputTxn

    outputTxn = OutputTransactionController.outputTransactionController cycleCounter headIdx
                  OutputTransactionController.OutputTransactionIn
                    { otcAllDone       = RowComputeUnit.rcAllDone compute
                    , otcConsumeSignal = consumeSignal
                    }

    effectiveRowIndex :: Signal dom (Index HeadDimension)
    effectiveRowIndex = mux (RowScheduler.rsOutputValid rsIn .&&. RowScheduler.rsConsumeSignal rsIn)
                            (pure 0)
                            rowIndex

    weightFetch = WeightFetchUnit.weightFetchUnit cycleCounter dramSlaveIn layerIdx headIdx
                    WeightFetchUnit.WeightFetchIn
                      { wfRowIndex      = effectiveRowIndex
                      , wfRowReqValid   = RowComputeUnit.rcFetchReq compute
                      , wfConsumeSignal = consumeSignal
                      , wfRowDone       = RowComputeUnit.rcRowDone compute
                      , wfInputValid    = inputValid
                      }

    currentRowDram = WeightFetchUnit.wfWeightDram weightFetch
    weightValid    = WeightFetchUnit.wfWeightValid weightFetch
    weightReady    = WeightFetchUnit.wfIdleReady weightFetch

    justConsumed :: Signal dom Bool
    justConsumed = register False consumeSignal

    effectiveInputValid = inputValidLatched .&&.
                          (not <$> OutputTransactionController.otcOutputValid outputTxn) .&&.
                          (not <$> justConsumed)

    -- xNorm BRAM: written by rmsNorm before computation starts, read serially
    -- by the RowComputeUnit one element per cycle (1-cycle BRAM latency).
    xNormRdData :: Signal dom FixedPoint
    xNormRdData = blockRam (repeat 0 :: Vec ModelDimension FixedPoint)
                    (RowComputeUnit.rcColumnAddr compute)
                    xNormWrite

    compute = RowComputeUnit.rowComputeUnit cycleCounter
            RowComputeUnit.RowComputeIn
              { rcInputValid      = effectiveInputValid
              , rcWeightValid     = weightValid
              , rcDownStreamReady = downStreamReady
              , rcRowIndex        = rowIndex
              , rcWeightDram      = currentRowDram
              , rcColumnRdData    = xNormRdData
              }

    readyForInput = RowComputeUnit.rcIdleReady compute .&&. weightReady

    rowDone   = RowComputeUnit.rcRowDone compute
    rowResult = RowComputeUnit.rcResult compute  -- scalar result for completed row

    -- -----------------------------------------------------------------------
    -- Rotary-encoded BRAM write logic
    -- Pairs: row 0,1 → pair 0; row 2,3 → pair 1; etc.
    -- On even rowDone: latch q_even.
    -- On odd  rowDone: compute rotary pair, write q'_even immediately,
    --                  buffer q'_odd for next cycle.
    -- -----------------------------------------------------------------------
    isOddRow :: Signal dom Bool
    isOddRow = odd . fromEnum <$> rowIndex

    -- Latch the even element of each pair
    qEvenLatched :: Signal dom FixedPoint
    qEvenLatched = regEn 0 (rowDone .&&. (not <$> isOddRow)) rowResult

    -- Pair index = rowIndex / 2
    pairIdx :: Signal dom (Index RotaryPositionalEmbeddingDimension)
    pairIdx = (\r -> toEnum (fromEnum r `div` 2)) <$> rowIndex

    cosVal :: Signal dom FixedPoint
    cosVal = (!!) <$> cosVec <*> pairIdx

    sinVal :: Signal dom FixedPoint
    sinVal = (!!) <$> sinVec <*> pairIdx

    -- Rotary-encoded even element:  q'[2i]   = q[2i]*c - q[2i+1]*s
    qEvenRotated :: Signal dom FixedPoint
    qEvenRotated = liftA2 (-) (liftA2 (*) qEvenLatched cosVal)
                              (liftA2 (*) rowResult     sinVal)

    -- Rotary-encoded odd element:   q'[2i+1] = q[2i]*s + q[2i+1]*c
    qOddRotated :: Signal dom FixedPoint
    qOddRotated  = liftA2 (+) (liftA2 (*) qEvenLatched sinVal)
                              (liftA2 (*) rowResult     cosVal)

    -- On the cycle of odd rowDone: write q'_even at (rowIndex-1)
    bramWriteImm :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
    bramWriteImm = mux (rowDone .&&. isOddRow)
                       ((\addr val -> Just (addr, val)) <$> (subtract 1 <$> rowIndex) <*> qEvenRotated)
                       (pure Nothing)

    -- One cycle later: write q'_odd at rowIndex (the odd address)
    pendingWrite :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
    pendingWrite = register Nothing $
                   mux (rowDone .&&. isOddRow)
                       ((\addr val -> Just (addr, val)) <$> rowIndex <*> qOddRotated)
                       (pure Nothing)

    -- Final BRAM write: immediate on odd rowDone cycle, pending on cycle after
    qBramWrite :: Signal dom (Maybe (Index HeadDimension, FixedPoint))
    qBramWrite = mux (rowDone .&&. isOddRow) bramWriteImm pendingWrite

    debugInfo = QHeadDebugInfo
      { qhRowIndex        = rowIndex
      , qhState           = RowComputeUnit.rcMultState compute
      , qhFirstMant       = register 0 (head . rowMantissas <$> currentRowDram)
      , qhRowResult       = register 0 rowResult
      , qhRowDone         = rowDone
      , qhFetchValid      = weightValid
      , qhFetchedWord     = pure 0
      , qhRowReset        = RowComputeUnit.rmdRowReset (RowComputeUnit.rcDebug compute)
      , qhRowEnable       = RowComputeUnit.rmdRowEnable (RowComputeUnit.rcDebug compute)
      , qhAccumValue      = RowComputeUnit.rmdAccValue (RowComputeUnit.rcDebug compute)
      , qhQOut            = pure (repeat 0)  -- replaced by BRAM write; kept for debug compat
      , qhCurrentRowExp   = register 0 (rowExponent <$> currentRowDram)
      , qhCurrentRowMant0 = register 0 (head . rowMantissas <$> currentRowDram)
      , qhRowReqValid     = RowComputeUnit.rcFetchReq compute
      , qhWeightReady     = weightReady
      , qhWeightValid     = weightValid
      }

--------------------------------------------------------------------------------
-- Top Level: queryHeadProjector
--------------------------------------------------------------------------------
queryHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumQueryHeads
  -> Signal dom Bool                                                -- inputValid
  -> Signal dom Bool                                                -- downStreamReady
  -> Signal dom Bool                                                -- consumeSignal
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint) -- cosVec
  -> Signal dom (Vec RotaryPositionalEmbeddingDimension FixedPoint) -- sinVec
  -> Signal dom (Maybe (Index ModelDimension, FixedPoint))          -- xNormWrite
  -> ( Master.AxiMasterOut dom
     , Signal dom (Maybe (Index HeadDimension, FixedPoint))         -- Q BRAM write
     , Signal dom Bool                                              -- outputValid
     , Signal dom Bool                                              -- readyForInput
     , QHeadDebugInfo dom
     )
queryHeadProjector cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal cosVec sinVec xNormWrite =
  ( qhcAxiMaster core
  , qhcBramWrite core
  , qhcOutputValid core
  , qhcReady core
  , qhcDebug core
  )
  where
    core = queryHeadCore cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal cosVec sinVec xNormWrite
