module LLaMa2.Memory.AXI.Master
(AxiMasterOut(..))
where

import Clash.Prelude
import qualified LLaMa2.Memory.AXI.Types as AXITypes (AxiAR, AxiAW, AxiW)

-- Complete AXI4 Master Interface (output signals from FPGA)
data AxiMasterOut dom = AxiMasterOut
  { -- Address Read Channel
    arvalid :: Signal dom Bool
  , ardata  :: Signal dom AXITypes.AxiAR
  
    -- Read Data Channel
  , rready  :: Signal dom Bool
  
    -- Address Write Channel
  , awvalid :: Signal dom Bool
  , awdata  :: Signal dom AXITypes.AxiAW
  
    -- Write Data Channel  
  , wvalid  :: Signal dom Bool
  , wdata   :: Signal dom AXITypes.AxiW
  
    -- Write Response Channel
  , bready  :: Signal dom Bool
  }
