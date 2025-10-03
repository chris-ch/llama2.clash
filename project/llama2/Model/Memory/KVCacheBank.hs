module Model.Memory.KVCacheBank (
    KVBank(..)
  , KVRamOwner(..)
  , makeRamOwnerKV
  , writeSequencer
) where

import Clash.Prelude

import Model.Core.Types ( TrueDualPortRunner )
import Model.Config
  ( BankDepth
  , HeadDimension
  , NumKeyValueHeads )
import Model.Config.KVGroups
  ( KVExpDepth )
import qualified Model.Memory.RamOps as RamOps (toRamOperation)

import Model.Numeric.Types
  ( Activation, Exponent )
 -- keep both, select by mode

-- A KV bank holds 4 RAMs:
--  - Mantissa BRAMs for K and V (depth = BankDepth = SeqLen * HeadDim)
--  - Exponent  BRAMs for K and V (depth = KVExpDepth = SeqLen * Groups)
data KVBank dom = KVBank
  { runKeyMantBank   :: TrueDualPortRunner dom BankDepth Activation
  , runKeyExpBank    :: TrueDualPortRunner dom KVExpDepth Exponent
  , runValueMantBank :: TrueDualPortRunner dom BankDepth Activation
  , runValueExpBank  :: TrueDualPortRunner dom KVExpDepth Exponent
  }

-- True dual-port runner
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

-- Write sequencer: emits mant per cycle; exponent at each group start
writeSequencer :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom Bool
writeSequencer enSig = doneSig
 where
  -- Head-dim counter
  dimCnt :: Signal dom (Index HeadDimension)
  dimCnt     = register 0 nextDimCnt
  nextDimCnt :: Signal dom (Index HeadDimension)
  nextDimCnt = mux enSig (fmap (\d -> if d == maxBound then 0 else succ d) dimCnt) (pure 0)
  atLastDim  = (== maxBound) <$> dimCnt

  doneSig :: Signal dom Bool
  doneSig    = (&&) <$> enSig <*> atLastDim
