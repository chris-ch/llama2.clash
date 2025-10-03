module Model.Config.KVGroups
  ( KVExpGroupLg2
  , KVExpGroupSize
  , KVExpGroups
  , KVExpDepth
  , KVExpAddress
  , groupSizeI
  ) where

import Clash.Prelude
import qualified GHC.TypeNats as TN
import Data.Type.Bool (If, type (&&))

import Model.Config (HeadDimension, SequenceLength)

-- Type-level equality test for Nats
type family EqNat (a :: Nat) (b :: Nat) :: Bool where
  EqNat a a = 'True
  EqNat a b = 'False

-- Choose the LARGEST power-of-two in {16,8,4,2} that:
--   - divides HeadDimension, and
--   - yields at least 2 groups (i.e., Div HeadDimension size >= 2).
-- This guarantees: HeadDimension = KVExpGroups * KVExpGroupSize, and KVExpGroups >= 2 when HeadDimension >= 8.
type KVExpGroupLg2 =
    If ((16 <=? HeadDimension) && EqNat (Mod HeadDimension 16) 0 && (2 <=? Div HeadDimension 16)) 4
  ( If (( 8 <=? HeadDimension) && EqNat (Mod HeadDimension  8) 0 && (2 <=? Div HeadDimension  8)) 3
  ( If (( 4 <=? HeadDimension) && EqNat (Mod HeadDimension  4) 0 && (2 <=? Div HeadDimension  4)) 2
                                                                                          1))

type KVExpGroupSize = 2 ^ KVExpGroupLg2
type KVExpGroups    = Div HeadDimension KVExpGroupSize

-- Exponent-bank depth (per head): SequenceLength * KVExpGroups
type KVExpDepth   = SequenceLength TN.* KVExpGroups
type KVExpAddress = Index KVExpDepth

groupSizeI :: Int
groupSizeI = natToNum @KVExpGroupSize
