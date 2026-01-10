module LLaMa2.Layer.Attention.KeyValueHeadProjector
  ( -- * DRAM-backed interface (new)
    keyValueHeadProjectorDRAM
  , KVHeadDebugInfo(..)
    -- * Hardcoded interface (legacy, for backward compatibility)
  , keyValueHeadProjector
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, SequenceLength, NumLayers, NumKeyValueHeads )
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (MatI8E)
import LLaMa2.Layer.Attention.RotaryEncoding (rotaryEncoder)
import qualified Simulation.Parameters as PARAM

import qualified LLaMa2.Numeric.Operations as OPS
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master

import LLaMa2.Layer.Attention.KeyValueHeadProjector.KeyValueHeadCore 
  ( keyValueHeadCore, KVHeadDebugInfo(..) )

--------------------------------------------------------------------------------
-- DRAM-backed KV Head Projector (NEW)
--------------------------------------------------------------------------------

-- | KV head projector with DRAM weight loading
--
-- == Overview
--
-- This is the new DRAM-backed version that loads K and V weights from DRAM
-- instead of using hardcoded weights. It follows the same pattern as
-- 'queryHeadProjector' for Q weights.
--
-- == Interface Changes from Legacy
--
-- @
-- -- Legacy (hardcoded weights):
-- keyValueHeadProjector inputValid downStreamReady seqPos xHat kvHeadParams rotary
--
-- -- New (DRAM weights):
-- keyValueHeadProjectorDRAM cycleCounter dramSlaveIn layerIdx kvHeadIdx
--                           inputValid downStreamReady consumeSignal seqPos xHat params
-- @
--
-- == Architecture
--
-- @
--   xHat ───────────────────────────────────────────────────────┐
--   seqPos ────────────────────────────────────────────┐        │
--                                                      │        │
--                  ┌─────────────────────────────────┐ │        │
--   dramSlaveIn ──►│       KeyValueHeadCore          │ │        │
--                  │                                 │ │        │
--                  │  ┌──────────────────────────┐   │ │        │
--                  │  │   KVWeightFetchUnit      │   │ │        │
--                  │  │   (loads K+V from DRAM)  │   │ │        │
--                  │  └──────────────────────────┘   │ │        │
--                  │                                 │ │        │
--                  │  ┌──────────────────────────┐   │ │        │
--                  │  │   KVRowComputeUnit       │◄──┼─┼────────┘
--                  │  │   (K·xHat and V·xHat)    │   │ │
--                  │  └──────────────────────────┘   │ │
--                  │              │                  │ │
--                  └──────────────┼──────────────────┘ │
--                                 │                    │
--                                 ▼                    │
--                     ┌───────────────────┐            │
--                     │  Rotary Encoding  │◄───────────┘
--                     │  (K only)         │
--                     └────────┬──────────┘
--                              │
--                              ▼
--                        kWithRotary, vOut
-- @
--
-- == Usage
--
-- @
-- (axiMaster, kOut, vOut, outputValid, readyForInput, debug) =
--   keyValueHeadProjectorDRAM cycleCounter dramSlaveIn layerIdx kvHeadIdx
--                             inputValid downStreamReady consumeSignal seqPos xHat params
-- @
--
keyValueHeadProjectorDRAM :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)               -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom                   -- ^ AXI slave (from arbiter)
  -> Index NumLayers                        -- ^ Layer index (static)
  -> Index NumKeyValueHeads                 -- ^ KV head index (static)
  -> Signal dom Bool                        -- ^ inputValid
  -> Signal dom Bool                        -- ^ downStreamReady
  -> Signal dom Bool                        -- ^ consumeSignal (coordinated clearing)
  -> Signal dom (Index SequenceLength)      -- ^ seqPos (for rotary encoding)
  -> Signal dom (Vec ModelDimension FixedPoint)  -- ^ xHat (normalized input)
  -> PARAM.DecoderParameters                -- ^ Model parameters
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec HeadDimension FixedPoint)  -- ^ K output (with rotary)
     , Signal dom (Vec HeadDimension FixedPoint)  -- ^ V output
     , Signal dom Bool                             -- ^ outputValid
     , Signal dom Bool                             -- ^ readyForInput
     , KVHeadDebugInfo dom
     )
keyValueHeadProjectorDRAM cycleCounter dramSlaveIn layerIdx kvHeadIdx 
                          inputValid downStreamReady consumeSignal seqPos xHat params =
  ( axiMaster
  , kWithRotary
  , vOut
  , outputValid
  , readyForInput
  , debug
  )
  where
    -- Get rotary encoding parameters
    rotary = PARAM.rotaryEncoding params

    -- Core computation
    (axiMaster, kOut, vOut, outputValid, readyForInput, debug) =
      keyValueHeadCore cycleCounter dramSlaveIn layerIdx kvHeadIdx
                       inputValid downStreamReady consumeSignal xHat params

    -- Apply rotary encoding to K output only (V doesn't get rotary)
    kWithRotary = (rotaryEncoder rotary <$> seqPos) <*> kOut

--------------------------------------------------------------------------------
-- Legacy Hardcoded KV Head Projector (for backward compatibility)
--------------------------------------------------------------------------------

-- | KV head projector with hardcoded weights (LEGACY)
--
-- This is the original implementation using hardcoded weights.
-- Kept for backward compatibility during migration.
--
-- @
-- WARNING: This function will be deprecated once DRAM migration is complete.
-- Use 'keyValueHeadProjectorDRAM' for new code.
-- @
--
keyValueHeadProjector :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom Bool                              -- ^ inputValid
  -> Signal dom Bool                              -- ^ downStreamReady
  -> Signal dom (Index SequenceLength)            -- ^ seqPos
  -> Signal dom (Vec ModelDimension FixedPoint)   -- ^ xHat
  -> PARAM.KeyValueHeadComponentQ                 -- ^ KV head parameters (hardcoded weights)
  -> PARAM.RotaryEncodingComponentF               -- ^ Rotary encoding parameters
  -> ( Signal dom (Vec HeadDimension FixedPoint)  -- ^ K output (with rotary)
     , Signal dom (Vec HeadDimension FixedPoint)  -- ^ V output
     , Signal dom Bool                             -- ^ outputValid
     , Signal dom Bool                             -- ^ readyForInput
     )
keyValueHeadProjector inputValid downStreamReady stepCountSig xHatSig kvHeadParams rotary =
  (kRoOut, vOut, outputValid, readyForInput)
 where
  selectedK :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedK = pure (PARAM.kMatrix kvHeadParams)

  selectedV :: Signal dom (MatI8E HeadDimension ModelDimension)
  selectedV = pure (PARAM.vMatrix kvHeadParams)

  (kOut, kValidOut, kReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedK xHatSig

  (vOut, vValidOut, vReadyOut) =
    OPS.parallelRowMatrixMultiplierDyn inputValid downStreamReady selectedV xHatSig

  kRoOut = (rotaryEncoder rotary <$> stepCountSig) <*> kOut

  outputValid = kValidOut .&&. vValidOut
  readyForInput = kReadyOut .&&. vReadyOut
