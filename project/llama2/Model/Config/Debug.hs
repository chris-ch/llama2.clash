module Model.Config.Debug
  ( AttnMode(..)
  , attnMode
  , attnEps
  , rowCheckKV
  ) where

import Clash.Prelude
import Model.Numeric.Types (FixedPoint)

-- Progressive bring-up modes for Stage3 attention:
--   AttnBaseline        : original combinational attend over register mirror drives the layer.
--   AttnShadowBRAM      : baseline drives; also compute sequential BRAM(I8E)-streamed attention (for compare).
--   AttnReplaceBRAM_F   : replace Stage3 by streaming from BRAM rows that store UNQUANTIZED K/V (FixedPoint).
--   AttnReplaceBRAM_Q   : replace Stage3 by streaming from BRAM(I8E) with dequantization (final target).
data AttnMode = AttnBaseline | AttnShadowBRAM | AttnReplaceBRAMF | AttnReplaceBRAMQ
  deriving (Show, Eq)

-- Start progressively: BRAM-F replacement first (no quantization difference).
attnMode :: AttnMode
attnMode = AttnReplaceBRAMF

-- Tolerance for comparing baseline vs streamed head outputs (SFixed 12.20).
attnEps :: FixedPoint
attnEps = 2 ^^ (-12 :: Int)  -- about 2.4e-4

-- Gate KV row-diff checks.
-- In Q mode the per-row FixedPoint vs dequantized(I8E) mismatch is expected and large.
rowCheckKV :: Bool
rowCheckKV = case attnMode of
  AttnReplaceBRAMQ -> False
  _                -> True
