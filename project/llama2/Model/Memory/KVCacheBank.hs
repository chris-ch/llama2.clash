module Model.Memory.KVCacheBank (
    KVBank(..)
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
import Model.Numeric.Types (Activation, Exponent, FixedPoint, Mantissa)
import Model.Numeric.Fixed
    ( quantizeI8E, quantizeI8E, quantizeI8E_ceilSafe )
import Data.Maybe (fromMaybe)

import Model.Config.Quant (quantModeKV, QuantMode(..))
import qualified Model.Memory.Addressing as Addressing (computeBankAddress)
import qualified GHC.TypeNats as TN

-- A KV bank holds 4 RAMs:
--  - Mantissa BRAMs for K and V (depth = BankDepth = SeqLen * HeadDim)
--  - Exponent  BRAMs for K and V (depth = SequenceLength)
data KVBank dom = KVBank
  { runKeyMantBank   :: TrueDualPortRunner dom BankDepth Activation
  , runKeyExpBank    :: TrueDualPortRunner dom SequenceLength Exponent
  , runValueMantBank :: TrueDualPortRunner dom BankDepth Activation
  , runValueExpBank  :: TrueDualPortRunner dom SequenceLength Exponent
  }

-- True dual-port runner over arbitrary (n, a)
mkTrueDualPortRunner
  :: (HiddenClockResetEnable dom, KnownNat n, NFDataX a)
  => TrueDualPortRunner dom n a
mkTrueDualPortRunner (addressA, writeA) (addressB, writeB) =
  trueDualPortBlockRam (RamOps.toRamOperation addressA writeA)
                       (RamOps.toRamOperation addressB writeB)

makeBankKV :: HiddenClockResetEnable dom => KVBank dom
makeBankKV = KVBank
  { runKeyMantBank   = mkTrueDualPortRunner
  , runKeyExpBank    = mkTrueDualPortRunner
  , runValueMantBank = mkTrueDualPortRunner
  , runValueExpBank  = mkTrueDualPortRunner
  }

newtype KVRamOwner dom = KVRamOwner
  { kvBanks :: Vec NumKeyValueHeads (KVBank dom)
  }

makeRamOwnerKV :: HiddenClockResetEnable dom => KVRamOwner dom
makeRamOwnerKV = KVRamOwner { kvBanks = map (const makeBankKV) indicesI }

writeSequencer :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec HeadDimension FixedPoint, Vec HeadDimension FixedPoint)
  -> ( Signal dom BankAddress
     , Signal dom (Maybe (BankAddress, Activation))
     , Signal dom (Maybe (Index SequenceLength, Exponent))
     , Signal dom (Maybe (BankAddress, Activation))
     , Signal dom (Maybe (Index SequenceLength, Exponent))
     , Signal dom Bool)
writeSequencer enSig seqPosSig kvFixedSig =
  (bankAddrSig, kMantWr, kExpWr, vMantWr, vExpWr, doneSig)
 where
  dimCnt :: Signal dom (Index HeadDimension)
  dimCnt     = register 0 nextDimCnt
  atLastDim :: Signal dom Bool
  atLastDim  = (== maxBound) <$> dimCnt
  atFirstDim :: Signal dom Bool
  atFirstDim = (== 0) <$> dimCnt
  nextDimCnt :: Signal dom (Index HeadDimension)
  nextDimCnt = mux enSig (fmap (\d -> if d == maxBound then 0 else succ d) dimCnt) (pure 0)
  doneSig :: Signal dom Bool
  doneSig    = (&&) <$> enSig <*> atLastDim

  kF :: Signal dom (Vec HeadDimension FixedPoint)
  kF = fst <$> kvFixedSig

  vF :: Signal dom (Vec HeadDimension FixedPoint)
  vF = snd <$> kvFixedSig

  qRow :: Vec HeadDimension FixedPoint -> (Vec HeadDimension Activation, Exponent)
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
  
  mantAt :: Signal dom (Maybe (Vec HeadDimension Activation, Exponent)) -> Signal dom Activation
  mantAt qSig = ((\(mVec, _) d -> mVec !! d) . fromMaybe (repeat 0, 0) <$> qSig) <*> dimCnt

  kMantAt :: Signal dom Activation
  kMantAt = mantAt quantK

  vMantAt :: Signal dom Activation
  vMantAt = mantAt quantV

  bankAddrSig :: Signal dom (Index (SequenceLength TN.* HeadDimension))
  bankAddrSig = Addressing.computeBankAddress <$> seqPosSig <*> dimCnt

  kMantWr :: Signal dom (Maybe (Index (SequenceLength TN.* HeadDimension), Mantissa))
  kMantWr = mux enSig (Just <$> bundle (bankAddrSig, kMantAt)) (pure Nothing)
  vMantWr = mux enSig (Just <$> bundle (bankAddrSig, vMantAt)) (pure Nothing)

  kExpWr :: Signal dom (Maybe (Index SequenceLength, Exponent))
  kExpWr =
    mux ((&&) <$> enSig <*> atFirstDim)
        (Just <$> bundle (seqPosSig, snd . fromMaybe (repeat 0, 0) <$> quantK))
        (pure Nothing)

  vExpWr :: Signal dom (Maybe (Index SequenceLength, Exponent))
  vExpWr =
    mux ((&&) <$> enSig <*> atFirstDim)
        (Just <$> bundle (seqPosSig, snd . fromMaybe (repeat 0, 0) <$> quantV))
        (pure Nothing)
