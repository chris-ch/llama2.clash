module Model.Config.Debug
  ( AttnMode(..)
  , attnMode
  , attnEps
  ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint)

-- How Stage3 attention output is chosen:
--   AttnBaseline     : original register-mirror + combinational attendHead.
--   AttnStreamShadow : compute streamed attention in parallel for comparison,
--                      but still drive the layer with AttnBaseline.
--   AttnStreamReplace: use the streamed attention to drive the layer.
data AttnMode = AttnBaseline | AttnStreamShadow | AttnStreamReplace
  deriving (Show, Eq)

-- Set this to the mode you want during bring-up.
attnMode :: AttnMode
attnMode = AttnBaseline

-- Tolerance for comparing baseline vs streamed head outputs.
attnEps :: FixedPoint
attnEps = 2 ^^ (-12)  -- ~2.4e-4 in your SFixed 12.20
