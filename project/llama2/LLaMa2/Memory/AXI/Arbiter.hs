module LLaMa2.Memory.AXI.Arbiter
(axiArbiterWithRouting)
where

import Clash.Prelude

import LLaMa2.Memory.AXI.Types (AxiR(..), AxiB (..), AxiAW (..), AxiW (..))
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master

axiArbiterWithRouting :: forall dom n.
  (HiddenClockResetEnable dom, KnownNat n)
  => Signal dom (Unsigned 32)            -- ^ Cycle counter for tracing
  -> Slave.AxiSlaveIn dom                -- ^ Single DRAM slave
  -> Vec n (Master.AxiMasterOut dom)     -- ^ Multiple masters (heads)
  -> ( Master.AxiMasterOut dom           -- ^ Combined master to DRAM
     , Vec n (Slave.AxiSlaveIn dom)      -- ^ Per-head slave interfaces
     )
axiArbiterWithRouting cycleCounter slaveIn masters = (masterOut, perHeadSlaves)
  where
    arRequests :: Vec n (Signal dom Bool)
    arRequests = map Master.arvalid masters

    -- Transaction tracking state machine
    inFlight :: Signal dom Bool
    inFlight = register False nextInFlight

    transactionOwner :: Signal dom (Index n)
    transactionOwner = register 0 nextTransactionOwner

    lastGranted :: Signal dom (Index n)
    lastGranted = register 0 nextLastGranted

    -- Round-robin selection: find next requesting head
    nextRequester :: Signal dom (Index n)
    nextRequester = findNextRequester <$> bundle arRequests <*> lastGranted

    findNextRequester :: Vec n Bool -> Index n -> Index n
    findNextRequester reqs lastR =
      let start = if lastR == maxBound then 0 else lastR + 1
          go i cnt
            | cnt == (0 :: Int) = lastR
            | reqs !! i = i
            | i == maxBound = go 0 (cnt - 1)
            | otherwise = go (i + 1) (cnt - 1)
      in go start (natToNum @n)

    -- Active index: locked to owner when in-flight, otherwise round-robin
    activeIdx :: Signal dom (Index n)
    activeIdx = mux inFlight transactionOwner nextRequester

    -- Latch arvalid on first assertion to prevent drops
    requestLatched = register False nextRequestLatched
      where
        nextRequestLatched = mux arHandshake (pure False)
                          $ mux selectedArValid (pure True)
                            requestLatched

    -- AR handshake detection
    selectedArValid = (!!) <$> bundle arRequests <*> activeIdx
    arHandshake = (selectedArValid .||. requestLatched) .&&. Slave.arready slaveIn .&&. (not <$> inFlight)

    -- R channel handshake detection
    selectedRReady = (!!) <$> bundle (map Master.rready masters) <*> transactionOwner
    rHandshake = Slave.rvalid slaveIn .&&. selectedRReady
    rLast = rlast <$> Slave.rdata slaveIn
    transactionDone = rHandshake .&&. rLast .&&. inFlight

    -- State transitions
    nextInFlight = mux arHandshake (pure True)
                 $ mux transactionDone (pure False)
                   inFlight

    nextTransactionOwner = mux arHandshake activeIdx transactionOwner

    -- Update lastGranted only when a transaction completes (for fair round-robin)
    nextLastGranted = mux transactionDone transactionOwner lastGranted

    -- Build master output using activeIdx for AR, transactionOwner for R
    masterOut = Master.AxiMasterOut
      { arvalid = mux inFlight (pure False) selectedArValid
      , ardata  = (!!) <$> bundle (map Master.ardata masters) <*> activeIdx
      , rready  = (!!) <$> bundle (map Master.rready masters) <*> transactionOwner
      , awvalid = pure False
      , awdata  = pure (AxiAW 0 0 0 0 0)
      , wvalid  = pure False
      , wdata   = pure (AxiW 0 0 False)
      , bready  = pure False
      }

    -- Per-head slave interfaces with response routing
    perHeadSlaves :: Vec n (Slave.AxiSlaveIn dom)
    perHeadSlaves = map makeHeadSlave indicesI

    makeHeadSlave :: Index n -> Slave.AxiSlaveIn dom
    makeHeadSlave headIdx = Slave.AxiSlaveIn
      { arready = isActiveAndIdle .&&. Slave.arready slaveIn
      , rvalid  = isOwner .&&. Slave.rvalid slaveIn
      , rdata   = Slave.rdata slaveIn
      , awready = pure False
      , wready  = pure False
      , bvalid  = pure False
      , bdata   = pure (AxiB 0 0)
      }
      where
        isActiveAndIdle = (activeIdx .==. pure headIdx) .&&. (not <$> inFlight)
        isOwner = inFlight .&&. (transactionOwner .==. pure headIdx)
