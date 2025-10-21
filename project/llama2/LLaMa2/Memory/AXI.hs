module LLaMa2.Memory.AXI
(AxiMasterOut(..), AxiSlaveIn(..), AxiAR(..), AxiW(..), AxiAW(..), AxiR(..), AxiB(..))
where

import Clash.Prelude

-- AXI4 Read Address Channel
data AxiAR = AxiAR
  { araddr  :: Unsigned 32    -- Address
  , arlen   :: Unsigned 8     -- Burst length - 1 (0 = 1 transfer)
  , arsize  :: Unsigned 3     -- Transfer size (2^arsize bytes)
  , arburst :: Unsigned 2     -- Burst type (01 = INCR)
  , arid    :: Unsigned 4     -- Transaction ID
  } deriving (Generic, NFDataX, Show)

-- AXI4 Read Data Channel  
data AxiR = AxiR
  { rdata   :: BitVector 512  -- Read data (64 bytes)
  , rresp   :: Unsigned 2     -- Response (00 = OKAY)
  , rlast   :: Bool           -- Last transfer in burst
  , rid     :: Unsigned 4     -- Transaction ID
  } deriving (Generic, NFDataX, Show)

-- AXI4 Write Address Channel
data AxiAW = AxiAW
  { awaddr  :: Unsigned 32
  , awlen   :: Unsigned 8
  , awsize  :: Unsigned 3
  , awburst :: Unsigned 2
  , awid    :: Unsigned 4
  } deriving (Generic, NFDataX, Show)

-- AXI4 Write Data Channel
data AxiW = AxiW
  { wdata   :: BitVector 512  -- Write data
  , wstrb   :: BitVector 64   -- Write strobes (1 bit per byte)
  , wlast   :: Bool           -- Last transfer in burst
  } deriving (Generic, NFDataX, Show)

-- AXI4 Write Response Channel
data AxiB = AxiB
  { bresp   :: Unsigned 2
  , bid     :: Unsigned 4
  } deriving (Generic, NFDataX, Show)

-- Complete AXI4 Master Interface (output signals from FPGA)
data AxiMasterOut dom = AxiMasterOut
  { -- Address Read Channel
    arvalid :: Signal dom Bool
  , ardata  :: Signal dom AxiAR
  
    -- Read Data Channel
  , rready  :: Signal dom Bool
  
    -- Address Write Channel
  , awvalid :: Signal dom Bool
  , awdata  :: Signal dom AxiAW
  
    -- Write Data Channel  
  , wvalid  :: Signal dom Bool
  , wdataMI   :: Signal dom AxiW
  
    -- Write Response Channel
  , bready  :: Signal dom Bool
  }

-- Complete AXI4 Slave Interface (input signals to FPGA)
data AxiSlaveIn dom = AxiSlaveIn
  { -- Address Read Channel
    arready :: Signal dom Bool
  
    -- Read Data Channel
  , rvalid  :: Signal dom Bool
  , rdataSI   :: Signal dom AxiR
  
    -- Address Write Channel
  , awready :: Signal dom Bool
  
    -- Write Data Channel
  , wready  :: Signal dom Bool
  
    -- Write Response Channel
  , bvalid  :: Signal dom Bool
  , bdata   :: Signal dom AxiB
  }
