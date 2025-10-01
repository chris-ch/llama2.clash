module Model.Config.Quant
  ( QuantMode(..)
  , quantModeKV
  ) where
    
import Clash.Prelude

data QuantMode = QuantNearest | QuantCeilSafe
  deriving (Show, Eq)

-- Start with CeilSafe to avoid clipping while we validate.
quantModeKV :: QuantMode
quantModeKV = QuantCeilSafe
