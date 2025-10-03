module Model.Memory.KVCacheBank (
    KVBank(..)
  , KVRamOwner(..)
  , makeRamOwnerKV
  , writeSequencer
) where

import Clash.Prelude
import qualified GHC.TypeNats as TN
import Data.Maybe (fromMaybe)

import Model.Core.Types ( TrueDualPortRunner )
import Model.Config
  ( BankDepth, BankAddress
  , SequenceLength, HeadDimension
  , NumKeyValueHeads )
import Model.Config.KVGroups
  ( KVExpAddress, KVExpDepth, KVExpGroupLg2, KVExpGroupSize, KVExpGroups )
import qualified Model.Memory.Addressing as Addressing
  ( computeBankAddress, computeExpAddress )
import qualified Model.Memory.RamOps as RamOps (toRamOperation)

import Model.Numeric.Types
  ( Activation, Exponent, FixedPoint )
import Model.Numeric.Fixed
  ( quantizeI8E, quantizeI8E_ceilSafe, quantizeI8E_best3_noClip ) -- keep both, select by mode

import Model.Config.Quant (quantModeKV, QuantMode(..))

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

-- Grouped quantization helper: split row into KVExpGroups of size KVExpGroupSize
qRowGrouped
  :: Vec HeadDimension FixedPoint
  -> Vec KVExpGroups (Vec KVExpGroupSize Activation, Exponent)
qRowGrouped row =
  let
    groups :: Vec KVExpGroups (Vec KVExpGroupSize FixedPoint)
    groups = unconcat dGroup row

    qOne :: Vec KVExpGroupSize FixedPoint -> (Vec KVExpGroupSize Activation, Exponent)
    qOne xs = case quantModeKV of
                QuantNearest  -> quantizeI8E xs
                QuantCeilSafe -> quantizeI8E_best3_noClip xs  -- better than plain ceil-safe
  in map qOne groups
 where
  dGroup :: SNat KVExpGroupSize
  dGroup = SNat

-- Select mantissa for the current dim from grouped result
mantAt
  :: Vec KVExpGroups (Vec KVExpGroupSize Activation, Exponent)
  -> Index HeadDimension
  -> Activation
mantAt qG dIx =
  let gLg2 = natToNum @KVExpGroupLg2
      gIdx = toEnum (fromEnum dIx `shiftR` gLg2)       :: Index KVExpGroups
      oIdx = toEnum (fromEnum dIx .&. (natToNum @KVExpGroupSize - 1))
                      :: Index KVExpGroupSize
      (mVec, _) = qG !! gIdx
  in mVec !! oIdx

-- Select exponent for the current group
expAtGroup
  :: Vec KVExpGroups (Vec KVExpGroupSize Activation, Exponent)
  -> Index HeadDimension
  -> Exponent
expAtGroup qG dIx =
  let gLg2 = natToNum @KVExpGroupLg2
      gIdx = toEnum (fromEnum dIx `shiftR` gLg2) :: Index KVExpGroups
  in snd (qG !! gIdx)

-- Write sequencer: emits mant per cycle; exponent at each group start
writeSequencer :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec HeadDimension FixedPoint, Vec HeadDimension FixedPoint)
  -> ( Signal dom BankAddress
     , Signal dom (Maybe (BankAddress, Activation))
     , Signal dom (Maybe (KVExpAddress, Exponent))
     , Signal dom (Maybe (BankAddress, Activation))
     , Signal dom (Maybe (KVExpAddress, Exponent))
     , Signal dom Bool)
writeSequencer enSig seqPosSig kvFixedSig =
  (bankAddrSig, kMantWr, kExpWr, vMantWr, vExpWr, doneSig)
 where
  -- Head-dim counter
  dimCnt :: Signal dom (Index HeadDimension)
  dimCnt     = register 0 nextDimCnt
  nextDimCnt :: Signal dom (Index HeadDimension)
  nextDimCnt = mux enSig (fmap (\d -> if d == maxBound then 0 else succ d) dimCnt) (pure 0)
  atLastDim  = (== maxBound) <$> dimCnt
  atFirstDim = (== 0) <$> dimCnt

  -- Group start when low KVExpGroupLg2 bits are zero
  atGroupStart :: Signal dom Bool
  atGroupStart =
    let mask = natToNum @KVExpGroupSize - 1
    in (\d en -> en && ((fromEnum d .&. mask) == 0)) <$> dimCnt <*> enSig

  doneSig :: Signal dom Bool
  doneSig    = (&&) <$> enSig <*> atLastDim

  -- Current row (K,V) in FixedPoint
  kF = fst <$> kvFixedSig
  vF = snd <$> kvFixedSig

  -- Hold grouped quantization across the row
  qHoldG vRowSig =
    mealy
      (\st (en, first, row) ->
         let newQ = qRowGrouped row
             st'  = if en then (if first then Just newQ else st) else Nothing
         in  (st', st'))
      Nothing
      (bundle (enSig, atFirstDim, vRowSig))

  quantK  = qHoldG kF  -- Signal dom (Maybe (Vec KVExpGroups (Vec KVExpGroupSize Activation, Exponent)))
  quantV  = qHoldG vF

  -- Mant at current dim
  kMantAt = mantAt . fromMaybe (repeat (repeat 0, 0)) <$> quantK <*> dimCnt
  vMantAt = mantAt . fromMaybe (repeat (repeat 0, 0)) <$> quantV <*> dimCnt

  -- Addresses
  bankAddrSig :: Signal dom (Index (SequenceLength TN.* HeadDimension))
  bankAddrSig = Addressing.computeBankAddress <$> seqPosSig <*> dimCnt

  -- Mant writes (every enabled cycle)
  kMantWr = mux enSig (Just <$> bundle (bankAddrSig, kMantAt)) (pure Nothing)
  vMantWr = mux enSig (Just <$> bundle (bankAddrSig, vMantAt)) (pure Nothing)

  -- Exp writes (each group start)
  expAddrSig :: Signal dom KVExpAddress
  expAddrSig = Addressing.computeExpAddress <$> seqPosSig <*> dimCnt

  kExpAt = expAtGroup . fromMaybe (repeat (repeat 0, 0)) <$> quantK <*> dimCnt
  vExpAt = expAtGroup . fromMaybe (repeat (repeat 0, 0)) <$> quantV <*> dimCnt

  kExpWr = mux atGroupStart (Just <$> bundle (expAddrSig, kExpAt)) (pure Nothing)
  vExpWr = mux atGroupStart (Just <$> bundle (expAddrSig, vExpAt)) (pure Nothing)
