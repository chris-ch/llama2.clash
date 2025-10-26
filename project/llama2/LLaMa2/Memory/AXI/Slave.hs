module LLaMa2.Memory.AXI.Slave
(AxiSlaveIn(..))
where
    
import Clash.Prelude

import qualified LLaMa2.Memory.AXI.Types as AXITypes (AxiR, AxiB)

-- Complete AXI4 Slave Interface (input signals to FPGA)
data AxiSlaveIn dom = AxiSlaveIn
  { -- Address Read Channel
    arready :: Signal dom Bool
  
    -- Read Data Channel
  , rvalid  :: Signal dom Bool
  , rdata   :: Signal dom AXITypes.AxiR
  
    -- Address Write Channel
  , awready :: Signal dom Bool
  
    -- Write Data Channel
  , wready  :: Signal dom Bool
  
    -- Write Response Channel
  , bvalid  :: Signal dom Bool
  , bdata   :: Signal dom AXITypes.AxiB
  }
