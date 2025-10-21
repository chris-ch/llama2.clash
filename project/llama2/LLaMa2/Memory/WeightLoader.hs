module LLaMa2.Memory.WeightLoader where

import Clash.Prelude
import LLaMa2.Numeric.Quantization (RowI8E, MatI8E)
import LLaMa2.Types.ModelConfig
import LLaMa2.Types.Parameters (TransformerLayerComponent)

-- Interface to external storage (eMMC via AXI)
data WeightMemoryInterface dom = WeightMemoryInterface
  { -- Read request
    readAddr  :: Signal dom (Unsigned 32)  -- Address in eMMC
  , readEn    :: Signal dom Bool           -- Start read
  , readReady :: Signal dom Bool           -- Ready to accept request
    
    -- Read response  
  , readData  :: Signal dom (BitVector 512) -- 64 bytes per cycle
  , readValid :: Signal dom Bool            -- Data valid
  }

-- Load one layer's weights from eMMC into DDR4 cache
layerWeightLoader :: forall dom . HiddenClockResetEnable dom
  => WeightMemoryInterface dom              -- eMMC interface
  -> Signal dom (Index NumLayers)           -- Which layer to load
  -> Signal dom TransformerLayerComponent   -- Output: loaded weights
layerWeightLoader = undefined
