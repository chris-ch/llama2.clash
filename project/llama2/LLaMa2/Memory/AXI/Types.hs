module LLaMa2.Memory.AXI.Types
( AxiAR(..), AxiW(..), AxiAW(..), AxiR(..), AxiB(..))
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
