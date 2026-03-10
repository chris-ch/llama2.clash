module LLaMa2.Layer.Attention.KVCache (
    kvBankControllerDRAM
) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumQueryHeads, HeadDimension, NumKeyValueHeads, SequenceLength, NumLayers)
import LLaMa2.Types.LayerData (LayerData (..))
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
  Index NumLayers                       ->
  Signal dom (Index SequenceLength)     ->
  Signal dom LayerData                  ->
  Signal dom Bool                       ->  -- ^ qkvValid
  Signal dom Bool                       ->  -- ^ enableWriteKV
  Signal dom Bool                       ->  -- ^ enableAttend
  Index NumKeyValueHeads                ->
  ( Master.AxiMasterOut dom
  , Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
  , Vec NumQueryHeads (Signal dom Bool)
  , Signal dom Bool                          -- ^ this bank's write done
  )
kvBankControllerDRAM cycleCounter dramSlaveIn layerIdx seqPos layerData
                     qkvValid enableWriteKV enableAttend kvIx =
  (axiMaster, headOutputs, headDones, wrDone)
  where
    keyVec  = (\ld -> keyVectors   ld !! kvIx) <$> layerData
    valVec  = (\ld -> valueVectors ld !! kvIx) <$> layerData
    queries = imap (\qIx _ -> (\ld -> queryVectors ld !! qIx) <$> layerData) (repeat ())

    (axiMaster, bankOutputs, bankDones, wrDone) =
      kvCacheBankController
        cycleCounter dramSlaveIn layerIdx kvIx
        seqPos qkvValid enableWriteKV enableAttend
        keyVec valVec queries

    -- headOutputs / headDones: bankOutputs already has zeros at non-bank indices
    headOutputs = bankOutputs
    headDones   = bankDones
