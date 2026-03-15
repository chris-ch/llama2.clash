module LLaMa2.Memory.AXI.Master
( AxiMasterOut(..)
, axiMasterMux
) where

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

-- | Field-by-field Bool mux of two AXI master outputs.
-- When sel=True, master 'a' drives the bus; when sel=False, master 'b' drives.
axiMasterMux :: forall dom. Signal dom Bool
             -> AxiMasterOut dom
             -> AxiMasterOut dom
             -> AxiMasterOut dom
axiMasterMux sel a b = AxiMasterOut
  { arvalid = mux sel (arvalid a) (arvalid b)
  , ardata  = mux sel (ardata  a) (ardata  b)
  , rready  = mux sel (rready  a) (rready  b)
  , awvalid = mux sel (awvalid a) (awvalid b)
  , awdata  = mux sel (awdata  a) (awdata  b)
  , wvalid  = mux sel (wvalid  a) (wvalid  b)
  , wdata   = mux sel (wdata   a) (wdata   b)
  , bready  = mux sel (bready  a) (bready  b)
  }
