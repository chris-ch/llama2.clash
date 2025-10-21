module LLaMa2.Memory.FakeAxiSlave (
  createFakeAxiSlave
) where

import Clash.Prelude
import LLaMa2.Memory.AXI

-- Creates a fake AXI slave that always responds "ready" but never returns data
-- Useful for testing that AXI infrastructure compiles and doesn't break existing code
createFakeAxiSlave :: AxiSlaveIn dom
createFakeAxiSlave = AxiSlaveIn
  { arready = pure True   -- Always ready to accept read address
  , rvalid  = pure False  -- Never returns read data (simulation only)
  , rdataSI = pure (AxiR 0 0 False 0)  -- Dummy data
  , awready = pure True   -- Always ready to accept write address
  , wready  = pure True   -- Always ready to accept write data
  , bvalid  = pure False  -- Never returns write response
  , bdata   = pure (AxiB 0 0)  -- Dummy response
  }
