-- ===== project/llama2/Model/Memory/KVCacheBank.hs =====
module Model.Memory.KVCacheBank (
    KvBank(..)
  , KVRamOwner(..)
  , makeRamOwnerKV
  , writeSequencer
) where

import Clash.Prelude
import qualified Prelude as P

import Model.Core.Types
  ( TrueDualPortRunner, BankDepth, BankAddress
  , NumKeyValueHeads, HeadDimension, SequenceLength )
import qualified Model.Memory.RamOps as RamOps (toRamOperation)
import qualified Model.Memory.Addressing as Addressing
import Model.Numeric.Types (Act, ExpS, F)
import Model.Numeric.Fixed (quantizeI8E)
import Data.Maybe (fromMaybe)

-- A KV bank holds 4 RAMs:
--  - Mantissa BRAMs for K and V (depth = BankDepth = SeqLen * HeadDim)
--  - Exponent  BRAMs for K and V (depth = SequenceLength)
data KvBank dom = KvBank
  { runKeyMantBank   :: TrueDualPortRunner dom BankDepth Act
  , runKeyExpBank    :: TrueDualPortRunner dom SequenceLength ExpS
  , runValueMantBank :: TrueDualPortRunner dom BankDepth Act
  , runValueExpBank  :: TrueDualPortRunner dom SequenceLength ExpS
  }

-- True dual-port runner over arbitrary (n, a)
mkTrueDualPortRunner
  :: (HiddenClockResetEnable dom, KnownNat n, NFDataX a)
  => TrueDualPortRunner dom n a
mkTrueDualPortRunner (addressA, writeA) (addressB, writeB) =
  trueDualPortBlockRam (RamOps.toRamOperation addressA writeA)
                       (RamOps.toRamOperation addressB writeB)

makeBankKV :: HiddenClockResetEnable dom => KvBank dom
makeBankKV = KvBank
  { runKeyMantBank   = mkTrueDualPortRunner
  , runKeyExpBank    = mkTrueDualPortRunner
  , runValueMantBank = mkTrueDualPortRunner
  , runValueExpBank  = mkTrueDualPortRunner
  }

newtype KVRamOwner dom = KVRamOwner
  { kvBanks :: Vec NumKeyValueHeads (KvBank dom)
  }

makeRamOwnerKV :: HiddenClockResetEnable dom => KVRamOwner dom
makeRamOwnerKV = KVRamOwner { kvBanks = map (const makeBankKV) indicesI }

-- Writer sequencer for one bank: generates addresses/writes and a done pulse.
-- Inputs are Float vectors (existing pipeline). We convert to F and quantize.
writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool                                     -- enable
  -> Signal dom (Index SequenceLength)                   -- sequence position
  -> Signal dom (Vec HeadDimension Float, Vec HeadDimension Float) -- (K, V) Float
  -> ( Signal dom BankAddress
     , Signal dom (Maybe (BankAddress, Act))             -- K mant write
     , Signal dom (Maybe (Index SequenceLength, ExpS))   -- K exp (row) write
     , Signal dom (Maybe (BankAddress, Act))             -- V mant write
     , Signal dom (Maybe (Index SequenceLength, ExpS))   -- V exp (row) write
     , Signal dom Bool)                                  -- done (1-cycle on last d)
writeSequencer enSig seqPosSig kvFloatSig =
  (bankAddrSig, kMantWr, kExpWr, vMantWr, vExpWr, doneSig)
 where
  -- Local dimension counter
  dimCnt    = register (0 :: Index HeadDimension) nextDimCnt
  atLastDim = (== maxBound) <$> dimCnt
  atFirstDim = (== (0 :: Index HeadDimension)) <$> dimCnt

  nextDimCnt =
    mux enSig
        (P.fmap (\d -> if d == maxBound then 0 else succ d) dimCnt)
        (pure 0)

  doneSig = (&&) <$> enSig <*> atLastDim

  -- Compute quantization once per row (when en && firstDim)
  kf = map realToFrac . fst <$> kvFloatSig
  vf = map realToFrac . snd <$> kvFloatSig

  (quantK, quantV) = (q kf, q vf)
   where
    q vF =
      let computed = quantizeI8E <$> vF
          latch s (go, v) =
            let fire = go && v
            in fire
          -- State machine: latch Just on en && firstDim; hold while en; clear when !en
      in mealy
           (\st (en, first, kvq) ->
              let st' = if en
                           then if first then Just kvq else st
                           else Nothing
              in  (st', st'))
           Nothing
           (bundle (enSig, atFirstDim, computed))

  -- Element at dim index
  mantAt qSig = ((\(mVec, _) d -> mVec !! d) P.. fromMaybe (repeat 0, 0) P.<$> qSig) <*> dimCnt

  expRow = fmap (fmap snd) quantK -- any Just carries the exponent; emitted only when firstDim

  kMantAt = mantAt quantK
  vMantAt = mantAt quantV

  -- Bank address for mantissas: time * HeadDim + dim
  bankAddrSig = Addressing.computeBankAddress <$> seqPosSig <*> dimCnt

  kMantWr =
    mux enSig (Just <$> bundle (bankAddrSig, kMantAt)) (pure Nothing)
  vMantWr =
    mux enSig (Just <$> bundle (bankAddrSig, vMantAt)) (pure Nothing)

  -- Row exponent writes: do it when en && firstDim
  kExpWr =
    mux ((&&) <$> enSig <*> atFirstDim)
        (Just <$> bundle (seqPosSig, snd . fromMaybe (repeat 0, 0) <$> quantK))
        (pure Nothing)

  vExpWr =
    mux ((&&) <$> enSig <*> atFirstDim)
        (Just <$> bundle (seqPosSig, snd . fromMaybe (repeat 0, 0) <$> quantV))
        (pure Nothing)
