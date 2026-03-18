module LLaMa2.Memory.AXI.Arbiter
(axiArbiterWithRouting)
where

import Clash.Prelude

import LLaMa2.Memory.AXI.Types (AxiR(..), AxiB (..), AxiAW (..), AxiW (..))
import qualified LLaMa2.Memory.AXI.Slave as Slave (AxiSlaveIn(..))
import qualified LLaMa2.Memory.AXI.Master as Master (AxiMasterOut(..))

-- | Round-robin arbiter priority selector.
-- For each head, computes its circular distance from the start position, then
-- folds over a priority-encoded Vec to find the earliest requester.
-- Uses fold (comparator tree) instead of recursion to avoid Clash specialization limits.
{-# NOINLINE roundRobinGrant #-}
roundRobinGrant :: forall n. KnownNat n => Vec n Bool -> Index n -> Index n
roundRobinGrant reqs lastR =
  snd (foldl pick (nW, lastR) (imap mkEntry reqs))
 where
  start = if lastR == maxBound then 0 else lastR + 1
  nW    = natToNum @n :: Unsigned 16
  -- Circular priority: distance from start position (lower = sooner in round-robin)
  prio :: Index n -> Unsigned 16
  prio i =
    let i16 = fromIntegral i :: Unsigned 16
        s16 = fromIntegral start :: Unsigned 16
    in if i16 >= s16 then i16 - s16 else nW - s16 + i16
  -- Requesting: (priority, index). Non-requesting: sentinel (nW, lastR).
  mkEntry i req = if req then (prio i, i) else (nW, lastR)
  pick (p1, i1) (p2, i2) = if p1 <= p2 then (p1, i1) else (p2, i2)

axiArbiterWithRouting :: forall dom n.
  (HiddenClockResetEnable dom, KnownNat n)
  => Slave.AxiSlaveIn dom                -- ^ Single DRAM slave
  -> Vec n (Master.AxiMasterOut dom)     -- ^ Multiple masters (heads)
  -> ( Master.AxiMasterOut dom           -- ^ Combined master to DRAM
     , Vec n (Slave.AxiSlaveIn dom)      -- ^ Per-head slave interfaces
     )
axiArbiterWithRouting slaveIn masters = (masterOut, perHeadSlaves)
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
    nextRequester = roundRobinGrant <$> bundle arRequests <*> lastGranted

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
      , awdata  = pure (AxiAW { awaddr = 0, awlen = 0, awsize = 0, awburst = 0, awid = 0 })
      , wvalid  = pure False
      , wdata   = pure (AxiW { wdata = 0, wstrb = 0, wlast = False })
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
      , bdata   = pure (AxiB { bresp = 0, bid = 0 })
      }
      where
        isActiveAndIdle = (activeIdx .==. pure headIdx) .&&. (not <$> inFlight)
        isOwner = inFlight .&&. (transactionOwner .==. pure headIdx)
