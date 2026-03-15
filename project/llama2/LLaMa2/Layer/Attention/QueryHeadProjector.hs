module LLaMa2.Layer.Attention.QueryHeadProjector
  ( queryHeadProjector
  , QHeadDebugInfo(..)
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
import LLaMa2.Numeric.Types (FixedPoint, Mantissa, Exponent)
import LLaMa2.Numeric.Quantization (RowI8E (..))
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryPositionEncoder)

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputTransactionController as OutputTransactionController
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.OutputAccumulator as OutputAccumulator
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
--------------------------------------------------------------------------------
data QueryHeadCoreOut dom = QueryHeadCoreOut
  { qhcAxiMaster   :: Master.AxiMasterOut dom
  , qhcResult      :: Signal dom (Vec HeadDimension FixedPoint)
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
  -> Signal dom (Vec ModelDimension FixedPoint)   -- xHat
  -> QueryHeadCoreOut dom
queryHeadCore cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat =
  QueryHeadCoreOut
    { qhcAxiMaster   = WeightFetchUnit.wfAxiMaster weightFetch
    , qhcResult      = qOut
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

    compute = RowComputeUnit.rowComputeUnit cycleCounter
            RowComputeUnit.RowComputeIn
              { rcInputValid      = effectiveInputValid
              , rcWeightValid     = weightValid
              , rcDownStreamReady = downStreamReady
              , rcRowIndex        = rowIndex
              , rcWeightDram      = currentRowDram
              , rcColumn          = xHat
              }

    readyForInput = RowComputeUnit.rcIdleReady compute .&&. weightReady

    rowDone = RowComputeUnit.rcRowDone compute

    outputAccum = OutputAccumulator.outputAccumulator cycleCounter headIdx
                    OutputAccumulator.OutputAccumIn
                      { oaRowDone   = RowComputeUnit.rcRowDone compute
                      , oaRowIndex  = rowIndex
                      , oaRowResult = RowComputeUnit.rcResult compute
                      }

    qOut = OutputAccumulator.oaOutput outputAccum

    debugInfo = QHeadDebugInfo
      { qhRowIndex        = rowIndex
      , qhState           = RowComputeUnit.rcMultState compute
      , qhFirstMant       = register 0 (head . rowMantissas <$> currentRowDram)
      , qhRowResult       = register 0 (RowComputeUnit.rcResult compute)
      , qhRowDone         = RowComputeUnit.rcRowDone compute
      , qhFetchValid      = weightValid
      , qhFetchedWord     = pure 0
      , qhRowReset        = RowComputeUnit.rmdRowReset (RowComputeUnit.rcDebug compute)
      , qhRowEnable       = RowComputeUnit.rmdRowEnable (RowComputeUnit.rcDebug compute)
      , qhAccumValue      = RowComputeUnit.rmdAccValue (RowComputeUnit.rcDebug compute)
      , qhQOut            = qOut
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
  -> Signal dom (Vec ModelDimension FixedPoint)                     -- xHat
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)
     , Signal dom Bool
     , Signal dom Bool
     , QHeadDebugInfo dom
     )
queryHeadProjector cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal cosVec sinVec xHat =
  ( qhcAxiMaster core
  , qWithRotary
  , qhcOutputValid core
  , qhcReady core
  , qhcDebug core
  )
  where
    core = queryHeadCore cycleCounter dramSlaveIn layerIdx headIdx inputValid downStreamReady consumeSignal xHat
    qWithRotary = rotaryPositionEncoder <$> qhcResult core <*> cosVec <*> sinVec
