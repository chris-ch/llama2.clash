module LLaMa2.Decoder.LayerStack (
  activeLayerProcessor, layerInputStage, LayerOutputs(..)
) where

import Clash.Prelude
import LLaMa2.Types.LayerData (LayerData(..))
import LLaMa2.Types.ModelConfig (NumLayers, NumKeyValueHeads, SequenceLength, ModelDimension, HeadDimension)
import qualified LLaMa2.Layer.TransformerLayer as TransformerLayer (transformerLayer)
import LLaMa2.Numeric.Types (FixedPoint)
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Types as AXITypes
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec)

data LayerOutputs dom = LayerOutputs
  { axiMasterOut    :: Master.AxiMasterOut dom              -- single weights AXI master
  , kvAxiMasterOuts :: Vec NumLayers (Vec NumKeyValueHeads (Master.AxiMasterOut dom))
  , qkvOutput       :: Signal dom LayerData
  , attnOutput      :: Signal dom LayerData
  , ffnOutput       :: Signal dom LayerData
  , writeDone       :: Signal dom Bool
  , attnDone        :: Signal dom Bool
  , qkvDone         :: Signal dom Bool
  , ffnDone         :: Signal dom Bool
  , qkvReady        :: Signal dom Bool
  }

activeLayerProcessor :: forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  )
  =>  Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom                                           -- weights DRAM interface
  -> Vec NumLayers (Vec NumKeyValueHeads (Slave.AxiSlaveIn dom))   -- KV cache DRAM per layer
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index SequenceLength)
  -> Signal dom LayerData
  -> Signal dom Bool
  -> LayerOutputs dom
activeLayerProcessor cycleCounter dramSlaveIn kvDramSlavesPerLayer activeLayerIdx seqPos inputData inputValid =
  LayerOutputs
    { axiMasterOut    = singleAxiMaster
    , kvAxiMasterOuts = distributeKvMasters activeLayerIdx singleKvMasters
    , qkvOutput       = qkvData
    , attnOutput      = attnData
    , ffnOutput       = ffnData
    , writeDone       = writeDone'
    , attnDone        = attnDone'
    , qkvDone         = qkvDone'
    , ffnDone         = ffnDone'
    , qkvReady        = qkvReady'
    }
  where
    -- Select KV slaves for the currently active layer
    kvDramSlaves :: Vec NumKeyValueHeads (Slave.AxiSlaveIn dom)
    kvDramSlaves = imap (\kvIx _ ->
        selectSlave activeLayerIdx (map (!! kvIx) kvDramSlavesPerLayer)
      ) (repeat ())

    -- Single transformer layer instance reused across all layer passes
    (singleAxiMaster, singleKvMasters, qProj, kProj, vProj, attnOut, ffnOut
      , qkvDone', writeDone', attnDone', ffnDone', qkvReady', _, _, _) =
      TransformerLayer.transformerLayer
        cycleCounter dramSlaveIn kvDramSlaves activeLayerIdx seqPos inputData inputValid

    qkvData  = (\d q k v -> d { queryVectors = q, keyVectors = k, valueVectors = v })
                  <$> inputData <*> qProj <*> kProj <*> vProj
    attnData = (\d attn -> d { attentionOutput = attn }) <$> inputData <*> attnOut
    ffnData  = (\d ffn  -> d { feedForwardOutput = ffn }) <$> inputData <*> ffnOut

-- | Mux a vector of slave inputs based on a dynamic index.
selectSlave :: forall dom n.
  KnownNat n
  => Signal dom (Index n)
  -> Vec n (Slave.AxiSlaveIn dom)
  -> Slave.AxiSlaveIn dom
selectSlave idx slaves = Slave.AxiSlaveIn
  { Slave.arready = sel Slave.arready
  , Slave.rvalid  = sel Slave.rvalid
  , Slave.rdata   = sel Slave.rdata
  , Slave.awready = sel Slave.awready
  , Slave.wready  = sel Slave.wready
  , Slave.bvalid  = sel Slave.bvalid
  , Slave.bdata   = sel Slave.bdata
  }
  where
    sel :: forall a. (Slave.AxiSlaveIn dom -> Signal dom a) -> Signal dom a
    sel field = (!!) <$> sequenceA (map field slaves) <*> idx

-- | Fan out KV masters to all per-layer slots; only the active layer's slot is live.
distributeKvMasters :: forall dom.
  Signal dom (Index NumLayers)
  -> Vec NumKeyValueHeads (Master.AxiMasterOut dom)
  -> Vec NumLayers (Vec NumKeyValueHeads (Master.AxiMasterOut dom))
distributeKvMasters activeLayerIdx kvMasters =
  imap (\layerIx _ ->
    let isActive = activeLayerIdx .==. pure layerIx
    in map (\m -> Master.axiMasterMux isActive m idleMaster) kvMasters
  ) (repeat ())
  where
    idleMaster = Master.AxiMasterOut
      { Master.arvalid = pure False
      , Master.ardata  = pure (AXITypes.AxiAR 0 0 0 0 0)
      , Master.rready  = pure False
      , Master.awvalid = pure False
      , Master.awdata  = pure (AXITypes.AxiAW 0 0 0 0 0)
      , Master.wvalid  = pure False
      , Master.wdata   = pure (AXITypes.AxiW 0 0 False)
      , Master.bready  = pure False
      }

layerInputStage :: Index NumLayers
                  -> LayerData
                  -> Vec ModelDimension FixedPoint
                  -> LayerData
layerInputStage idx currentData embedding
  | idx == 0  = currentData { inputVector = embedding }
  | otherwise = currentData { inputVector = feedForwardOutput currentData }
