module Model.Memory.KVCacheBank (
    KvBank(..)
  , KVRamOwner(..)
  , makeRamOwnerKV
  , writeSequencer
) where

import Clash.Prelude
import qualified Prelude as P

import Model.Core.Types ( TrueDualPortRunner )
import Model.Config
  ( BankDepth, BankAddress
  , NumKeyValueHeads, HeadDimension, SequenceLength )
import qualified Model.Memory.RamOps as RamOps (toRamOperation)
import qualified Model.Memory.Addressing as Addressing
import Model.Numeric.Types (Act, ExpS, FixedPoint)
import Model.Numeric.Fixed
    ( quantizeI8E, quantizeI8E, quantizeI8E_ceilSafe )
import Data.Maybe (fromMaybe)

import Model.Config.Quant (quantModeKV, QuantMode(..))

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

writeSequencer
  :: HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec HeadDimension FixedPoint, Vec HeadDimension FixedPoint)
  -> ( Signal dom BankAddress
     , Signal dom (Maybe (BankAddress, Act))
     , Signal dom (Maybe (Index SequenceLength, ExpS))
     , Signal dom (Maybe (BankAddress, Act))
     , Signal dom (Maybe (Index SequenceLength, ExpS))
     , Signal dom Bool)
writeSequencer enSig seqPosSig kvFixedSig =
  (bankAddrSig, kMantWr, kExpWr, vMantWr, vExpWr, doneSig)
 where

  dimCnt     = register (0 :: Index HeadDimension) nextDimCnt
  atLastDim  = (== maxBound) <$> dimCnt
  atFirstDim = (== (0 :: Index HeadDimension)) <$> dimCnt
  nextDimCnt = mux enSig (fmap (\d -> if d == maxBound then 0 else succ d) dimCnt) (pure 0)
  doneSig    = (&&) <$> enSig <*> atLastDim

  kF = fst <$> kvFixedSig
  vF = snd <$> kvFixedSig

  qRow :: Vec HeadDimension FixedPoint -> (Vec HeadDimension Act, ExpS)
  qRow xs = case quantModeKV of
              QuantNearest  -> quantizeI8E xs
              QuantCeilSafe -> quantizeI8E_ceilSafe xs

  (quantK, quantV) =
    ( qHold kF
    , qHold vF )
   where
    -- Hold one row’s quantization result across HeadDim cycles
    qHold vRowSig =
      mealy
        (\st (en, first, row) ->
           let newQ = qRow row
               st'  = if en then (if first then Just newQ else st) else Nothing
           in  (st', st'))
        Nothing
        (bundle (enSig, atFirstDim, vRowSig))

  mantAt qSig = ((\(mVec, _) d -> mVec !! d) . fromMaybe (repeat 0, 0) <$> qSig) <*> dimCnt
  kMantAt = mantAt quantK
  vMantAt = mantAt quantV

  bankAddrSig = Addressing.computeBankAddress <$> seqPosSig <*> dimCnt

  kMantWr = mux enSig (Just <$> bundle (bankAddrSig, kMantAt)) (pure Nothing)
  vMantWr = mux enSig (Just <$> bundle (bankAddrSig, vMantAt)) (pure Nothing)

  kExpWr =
    mux ((&&) <$> enSig <*> atFirstDim)
        (Just <$> bundle (seqPosSig, snd . fromMaybe (repeat 0, 0) <$> quantK))
        (pure Nothing)
  vExpWr =
    mux ((&&) <$> enSig <*> atFirstDim)
        (Just <$> bundle (seqPosSig, snd . fromMaybe (repeat 0, 0) <$> quantV))
        (pure Nothing)
