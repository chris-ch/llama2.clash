module Model.Layers.Attention.MultiHeadAttention (
    MultiHeadAttentionComponent(..)
) where

import Clash.Prelude
import qualified Prelude as P
import Model.Core.Types
  ( NumQueryHeads, ModelDimemsion, NumKeyValueHeads
  , HeadDimension, CArray2D (..), SingleHeadComponent (..)
  , SequenceLength)

data MultiHeadAttentionComponent = MultiHeadAttentionComponent
  { heads  :: Vec NumQueryHeads SingleHeadComponent
  , mWo    :: Vec NumQueryHeads (CArray2D ModelDimemsion HeadDimension)
  , rmsAtt :: Vec ModelDimemsion Float
  } deriving (Show)
