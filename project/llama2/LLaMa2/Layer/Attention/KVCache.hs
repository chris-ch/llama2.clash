module LLaMa2.Layer.Attention.KVCache (
    kvBankControllerDRAM
) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (HeadDimension, NumKeyValueHeads, SequenceLength, NumLayers, QHeadsPerKVBank)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Memory.WeightsLayout (WordsPerFPVec)
import LLaMa2.Layer.Attention.KVCacheBankController (kvCacheBankController)
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.AXI.Slave as Slave

-- | DRAM-backed KV cache bank controller.
-- Each bank covers all query heads that share this KV head.
-- Non-overlapping head outputs are left at zero; MultiHeadAttention combines them.
kvBankControllerDRAM ::
  forall dom.
  ( HiddenClockResetEnable dom
  , KnownNat (WordsPerFPVec HeadDimension)
  ) =>
  Signal dom (Unsigned 32)              ->
  Slave.AxiSlaveIn dom                  ->  -- ^ dedicated per-bank KV DRAM slave
  Signal dom (Index NumLayers)          ->
  Signal dom (Index SequenceLength)     ->
  Signal dom (Maybe (Index HeadDimension, FixedPoint))  ->  -- ^ K element writes (streaming)
  Signal dom (Maybe (Index HeadDimension, FixedPoint))  ->  -- ^ V element writes (streaming)
  Vec QHeadsPerKVBank (Signal dom FixedPoint)           ->  -- ^ Q BRAM read data (1-cycle latency)
  Signal dom Bool                       ->  -- ^ qkvValid
  Signal dom Bool                       ->  -- ^ enableWriteKV
  Signal dom Bool                       ->  -- ^ enableAttend
  Index NumKeyValueHeads                ->
  ( Master.AxiMasterOut dom
  , Vec QHeadsPerKVBank (Signal dom (Vec HeadDimension FixedPoint)) -- ^ head outputs (local order)
  , Vec QHeadsPerKVBank (Signal dom Bool)                           -- ^ head done flags (local order)
  , Signal dom Bool                          -- ^ this bank's write done
  , Vec QHeadsPerKVBank (Signal dom (Index HeadDimension))  -- ^ Q BRAM read addresses
  )
kvBankControllerDRAM cycleCounter dramSlaveIn layerIdx seqPos
                     keyVec valVec qBramRdDatas
                     qkvValid enableWriteKV enableAttend kvIx =
  (axiMaster, headOutputs, headDones, wrDone, qBramRdAddrs)
  where
    (axiMaster, bankOutputs, bankDones, wrDone, qBramRdAddrs) =
      kvCacheBankController
        cycleCounter dramSlaveIn layerIdx kvIx
        seqPos qkvValid enableWriteKV enableAttend
        keyVec valVec qBramRdDatas

    -- headOutputs / headDones: bankOutputs already has zeros at non-bank indices
    headOutputs = bankOutputs
    headDones   = bankDones
