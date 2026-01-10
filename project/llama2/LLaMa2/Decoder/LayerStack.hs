module LLaMa2.Decoder.LayerStack (
  activeLayerProcessor, layerInputStage, LayerOutputs(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..))
import LLaMa2.Types.ModelConfig (NumLayers, SequenceLength, ModelDimension, HeadDimension)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types (FixedPoint, Mantissa)
import qualified Simulation.Parameters as PARAM (DecoderParameters (..))
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Layer.Attention.QueryHeadProjector.QueryHeadCore as QueryHeadCore (QHeadDebugInfo (..))
import LLaMa2.Numeric.Operations (MultiplierState)

data LayerOutputs dom = LayerOutputs
  { axiMasterOuts  :: Vec NumLayers (Master.AxiMasterOut dom)
  , qkvOutput      :: Signal dom LayerData
  , attnOutput     :: Signal dom LayerData
  , ffnOutput      :: Signal dom LayerData
  , writeDone      :: Signal dom Bool
  , attnDone       :: Signal dom Bool
  , qkvDone        :: Signal dom Bool
  , ffnDone        :: Signal dom Bool
  , qkvReady       :: Signal dom Bool
  , dbgRowIndex    :: Signal dom (Index HeadDimension)
  , dbgState       :: Signal dom MultiplierState
  , dbgFirstMant   :: Signal dom Mantissa
  , dbgRowResult   :: Signal dom FixedPoint
  , dbgRowDone     :: Signal dom Bool
  , dbgFetchValid  :: Signal dom Bool
  , dbgFetchedWord :: Signal dom (BitVector 512)
  }

activeLayerProcessor :: forall dom.
  HiddenClockResetEnable dom
  =>  Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom                     -- DRAM interface
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index SequenceLength)
  -> Signal dom LayerData
  -> Signal dom Bool
  -> PARAM.DecoderParameters
  -> LayerOutputs dom
activeLayerProcessor cycleCounter dramSlaveIn activeLayerIdx seqPos inputData inputValid params =
  LayerOutputs
    { axiMasterOuts  = layerAxiMasters
    , qkvOutput      = selectedQkvOutput
    , attnOutput     = selectedAttnOutput
    , ffnOutput      = selectedFfnOutput
    , writeDone      = selectedWriteDone
    , attnDone       = selectedAttnDone
    , qkvDone        = selectedQkvDone
    , ffnDone        = selectedFfnDone
    , qkvReady       = selectedQkvReady
    , dbgRowIndex    = selectedDbgRowIndex
    , dbgState       = selectedDbgState
    , dbgFirstMant   = selectedDbgFirstMant
    , dbgRowResult   = selectedDbgRowResult
    , dbgRowDone     = selectedDbgRowDone
    , dbgFetchValid  = selectedDbgFetchValid
    , dbgFetchedWord = selectedDbgFetchedWord
    }
  where
    -- Run all layers in parallel
    layerOutputs :: Vec NumLayers (Master.AxiMasterOut dom, Signal dom LayerData, Signal dom LayerData,
      Signal dom LayerData, Signal dom Bool, Signal dom Bool,
      Signal dom Bool, Signal dom Bool, Signal dom Bool, QueryHeadCore.QHeadDebugInfo dom)
    layerOutputs = map (layerPipeline inputData params) indicesI

    -- Extract AXI masters and other outputs
    layerAxiMasters :: Vec NumLayers (Master.AxiMasterOut dom)
    layerAxiMasters = map (\(axi, _, _, _, _, _, _, _, _, _) -> axi) layerOutputs

    -- Extract vectors of each signal type
    qkvOutputs :: Vec NumLayers (Signal dom LayerData)
    qkvOutputs = map (\(_, qkv, _, _, _, _, _, _, _, _) -> qkv) layerOutputs

    attnOutputs :: Vec NumLayers (Signal dom LayerData)
    attnOutputs = map (\(_, _, attn, _, _, _, _, _, _, _) -> attn) layerOutputs

    ffnOutputs :: Vec NumLayers (Signal dom LayerData)
    ffnOutputs = map (\(_, _, _, ffn, _, _, _, _, _, _) -> ffn) layerOutputs

    qkvDones :: Vec NumLayers (Signal dom Bool)
    qkvDones = map (\(_, _, _, _, qkvD, _, _, _, _, _) -> qkvD) layerOutputs

    writeDones :: Vec NumLayers (Signal dom Bool)
    writeDones = map (\(_, _, _, _, _, writeD, _, _, _, _) -> writeD) layerOutputs

    attnDones :: Vec NumLayers (Signal dom Bool)
    attnDones = map (\(_, _, _, _, _, _, attnD, _, _, _) -> attnD) layerOutputs

    ffnDones :: Vec NumLayers (Signal dom Bool)
    ffnDones = map (\(_, _, _, _, _, _, _, ffnD, _, _) -> ffnD) layerOutputs

    qkvReadys :: Vec NumLayers (Signal dom Bool)
    qkvReadys = map (\(_, _, _, _, _, _, _, _, qkvR, _) -> qkvR) layerOutputs

    dbgInfos :: Vec NumLayers (QueryHeadCore.QHeadDebugInfo dom)
    dbgInfos = map (\(_, _, _, _, _, _, _, _, _, dbg) -> dbg) layerOutputs

    -- Extract debug info fields into separate vectors
    dbgRowIndexes :: Vec NumLayers (Signal dom (Index HeadDimension))
    dbgRowIndexes = map QueryHeadCore.qhRowIndex dbgInfos

    dbgStates :: Vec NumLayers (Signal dom MultiplierState)
    dbgStates = map QueryHeadCore.qhState dbgInfos

    dbgFirstMants :: Vec NumLayers (Signal dom Mantissa)
    dbgFirstMants = map QueryHeadCore.qhFirstMant dbgInfos

    dbgRowResults :: Vec NumLayers (Signal dom FixedPoint)
    dbgRowResults = map QueryHeadCore.qhRowResult dbgInfos

    dbgRowDones :: Vec NumLayers (Signal dom Bool)
    dbgRowDones = map QueryHeadCore.qhRowDone dbgInfos

    dbgFetchValids :: Vec NumLayers (Signal dom Bool)
    dbgFetchValids = map QueryHeadCore.qhFetchValid dbgInfos

    dbgFetchedWords :: Vec NumLayers (Signal dom (BitVector 512))
    dbgFetchedWords = map QueryHeadCore.qhFetchedWord dbgInfos

    -- Select outputs for the active layer
    selectedQkvOutput :: Signal dom LayerData
    selectedQkvOutput = selectActive activeLayerIdx qkvOutputs

    selectedAttnOutput :: Signal dom LayerData
    selectedAttnOutput = selectActive activeLayerIdx attnOutputs

    selectedFfnOutput :: Signal dom LayerData
    selectedFfnOutput = selectActive activeLayerIdx ffnOutputs

    selectedQkvDone :: Signal dom Bool
    selectedQkvDone = selectActive activeLayerIdx qkvDones

    selectedWriteDone :: Signal dom Bool
    selectedWriteDone = selectActive activeLayerIdx writeDones

    selectedAttnDone :: Signal dom Bool
    selectedAttnDone = selectActive activeLayerIdx attnDones

    selectedFfnDone :: Signal dom Bool
    selectedFfnDone = selectActive activeLayerIdx ffnDones

    selectedQkvReady :: Signal dom Bool
    selectedQkvReady = selectActive activeLayerIdx qkvReadys

    -- Select debug signals for the active layer
    selectedDbgRowIndex :: Signal dom (Index HeadDimension)
    selectedDbgRowIndex = selectActive activeLayerIdx dbgRowIndexes

    selectedDbgState :: Signal dom MultiplierState
    selectedDbgState = selectActive activeLayerIdx dbgStates

    selectedDbgFirstMant :: Signal dom Mantissa
    selectedDbgFirstMant = selectActive activeLayerIdx dbgFirstMants

    selectedDbgRowResult :: Signal dom FixedPoint
    selectedDbgRowResult = selectActive activeLayerIdx dbgRowResults

    selectedDbgRowDone :: Signal dom Bool
    selectedDbgRowDone = selectActive activeLayerIdx dbgRowDones

    selectedDbgFetchValid :: Signal dom Bool
    selectedDbgFetchValid = selectActive activeLayerIdx dbgFetchValids

    selectedDbgFetchedWord :: Signal dom (BitVector 512)
    selectedDbgFetchedWord = selectActive activeLayerIdx dbgFetchedWords

    layerPipeline :: Signal dom LayerData
                  -> PARAM.DecoderParameters
                  -> Index NumLayers
                  -> ( Master.AxiMasterOut dom
                     , Signal dom LayerData
                     , Signal dom LayerData
                     , Signal dom LayerData
                     , Signal dom Bool
                     , Signal dom Bool
                     , Signal dom Bool
                     , Signal dom Bool
                     , Signal dom Bool
                     , QueryHeadCore.QHeadDebugInfo dom
                     )
    layerPipeline inputData' params' layerIdx =
      ( axiMaster, qkvData, attnData, ffnData
      , qkvDone', writeDone', attnDone', ffnDone', qkvReady, qHeadDebugInfo )
      where
        isThisLayer = activeLayerIdx .==. pure layerIdx
        validIn' = inputValid .&&. isThisLayer

        ( axiMaster, qProj, kProj, vProj, attnOut, ffnOut
          , qkvDone', writeDone', attnDone', ffnDone', qkvReady, qHeadDebugInfo, _, _ ,_ ) =
          TransformerLayer.transformerLayer
            cycleCounter
            dramSlaveIn
            layerIdx
            params'
            seqPos
            inputData'
            validIn'

        qkvData  = (\d q k v -> d { queryVectors = q, keyVectors = k, valueVectors = v })
                      <$> inputData' <*> qProj <*> kProj <*> vProj
        attnData = (\d attn -> d { attentionOutput = attn }) <$> inputData' <*> attnOut
        ffnData  = (\d ffn -> d { feedForwardOutput = ffn }) <$> inputData' <*> ffnOut

    -- Helper function to select from a vector of signals based on dynamic index
    selectActive :: Signal dom (Index NumLayers) -> Vec NumLayers (Signal dom a) -> Signal dom a
    selectActive idx vec = (!!) <$> sequenceA vec <*> idx

layerInputStage :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
layerInputStage idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
