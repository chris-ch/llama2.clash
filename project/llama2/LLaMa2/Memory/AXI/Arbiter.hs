module LLaMa2.Memory.AXI.Arbiter
(axiArbiterWithRouting)
where

import Clash.Prelude

import LLaMa2.Memory.AXI.Types (AxiR(..), AxiB (..), AxiAW (..), AxiW (..))
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master

-- | Round-robin AXI arbiter with per-master response routing.
--
-- == Overview
--
-- This arbiter multiplexes multiple AXI masters (query heads) onto a single
-- AXI slave (DRAM controller). It implements round-robin arbitration for
-- fairness and tracks in-flight transactions to route responses back to
-- the correct requesting master.
--
-- == CURRENT STATUS: DISABLED
--
-- __IMPORTANT__: This arbiter is instantiated in qkvProjector but NOT connected
-- to the data path. The current working implementation (qResults) passes
-- dramSlaveIn directly to all query heads, bypassing the arbiter completely.
--
-- The arbiter code (qResults' with perHeadSlaves) exists but is unused:
-- @
-- (axiMasterOut, perHeadSlaves) = axiArbiterWithRouting dramSlaveIn qAxiMasters
-- qResults' = imap (\headIdx _ ->
--     queryHeadProjector (perHeadSlaves !! headIdx) ...  -- NOT USED
--   ) ...
-- 
-- qResults = map (qHead params) indicesI  -- ACTUALLY USED
--   where
--     qHead params' headIdx = 
--       queryHeadProjector dramSlaveIn layerIdx headIdx ...  -- Direct connection
-- @
--
-- To enable the arbiter, replace qResults with qResults' in qkvProjector.
--
-- == Architecture
--
-- @
--                    ┌─────────────────────────────────────────────────────────┐
--                    │              axiArbiterWithRouting                      │
--                    │                                                         │
--   masters[0] ─────►│  ┌─────────────────────────────────────────────────┐    │
--     .arvalid       │  │                                                 │    │
--     .ardata        │  │            Request Arbitration                  │    │
--     .rready        │  │                                                 │    │
--                    │  │  ┌──────────┐    ┌──────────────┐               │    │
--   masters[1] ─────►│  │  │ Round-   │    │ Transaction  │               │    │
--     ...            │  │  │ Robin    │───►│   Tracker    │               │    │
--                    │  │  │ Selector │    │              │               │    │
--   masters[N] ─────►│  │  └──────────┘    │ - inFlight   │               │    │
--                    │  │       ▲          │ - owner      │               │    │
--                    │  │       │          │ - lastGrant  │               │    │
--                    │  │  arRequests      │ - reqLatched │               │    │
--                    │  │                  └──────────────┘               │    │
--                    │  │                         │                       │    │
--                    │  └─────────────────────────┼───────────────────────┘    │
--                    │                            │                            │
--                    │                            ▼                            │
--                    │  ┌─────────────────────────────────────────────────┐    │
--                    │  │              masterOut (to DRAM)                │    │
--                    │  │                                                 │───►│
--                    │  │  .arvalid = !inFlight && selectedArValid        │    │
--                    │  │  .ardata  = masters[activeIdx].ardata           │    │
--                    │  │  .rready  = masters[owner].rready               │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--   slaveIn ────────►│  ┌─────────────────────────────────────────────────┐    │
--     .arready       │  │              Response Routing                   │    │
--     .rvalid        │  │                                                 │    │
--     .rdata         │  │  Route responses to transaction owner only:     │    │
--                    │  │                                                 │    │
--                    │  │  perHeadSlaves[i].arready =                     │───►│ perHeadSlaves[0]
--                    │  │      (activeIdx == i) && !inFlight &&           │    │
--                    │  │      slaveIn.arready                            │───►│ perHeadSlaves[1]
--                    │  │                                                 │    │   ...
--                    │  │  perHeadSlaves[i].rvalid =                      │───►│ perHeadSlaves[N]
--                    │  │      (owner == i) && inFlight &&                │    │
--                    │  │      slaveIn.rvalid                             │    │
--                    │  │                                                 │    │
--                    │  │  perHeadSlaves[i].rdata = slaveIn.rdata         │    │
--                    │  │      (broadcast, but only owner sees rvalid)    │    │
--                    │  │                                                 │    │
--                    │  └─────────────────────────────────────────────────┘    │
--                    │                                                         │
--                    └─────────────────────────────────────────────────────────┘
-- @
--
-- == State Registers
--
-- [@inFlight@] __Bool__. True when a transaction is active (AR accepted, waiting for R).
--              Prevents new AR requests from being issued.
--              Set when AR handshake completes, cleared when R handshake with rlast.
--
-- [@transactionOwner@] __Index n__. Which master owns the current in-flight transaction.
--                      Latched on AR handshake, used to route R responses.
--                      Only valid when inFlight is True.
--
-- [@lastGranted@] __Index n__. Last master that completed a transaction.
--                 Used for round-robin fairness: next arbitration starts
--                 from (lastGranted + 1).
--                 Updated when transaction completes (not when granted).
--
-- [@requestLatched@] __Bool__. Latches the selected master's arvalid to prevent
--                    drops during multi-cycle arready waits.
--                    Set when selectedArValid becomes True, cleared on AR handshake.
--                    
--                    __IMPLEMENTATION BUG__: requestLatched is used in arHandshake
--                    detection but NOT included in masterOut.arvalid, creating
--                    an inconsistency. The handshake can complete using the latched
--                    request, but the output arvalid doesn't reflect it.
--                    
--                    This may cause spurious handshakes or missed transactions.
--
-- == Arbitration Logic
--
-- === Round-Robin Selection
--
-- @
-- findNextRequester :: Vec n Bool -> Index n -> Index n
-- findNextRequester reqs lastR =
--   -- Start searching from (lastGranted + 1)
--   -- Wrap around, find first requesting master
--   -- If none requesting, return lastR (no change)
-- @
--
-- === Active Index Selection
--
-- @
-- activeIdx = mux inFlight 
--                 transactionOwner  -- Locked while in-flight
--                 nextRequester     -- Round-robin when idle
-- @
--
-- === Handshake Detection
--
-- @
-- selectedArValid = arRequests !! activeIdx
-- requestLatched = latch selectedArValid (cleared on handshake)
-- 
-- arHandshake = (selectedArValid || requestLatched) && slaveIn.arready && !inFlight
-- rHandshake  = slaveIn.rvalid && masters[owner].rready
-- rLast       = slaveIn.rdata.rlast
-- transactionDone = rHandshake && rLast && inFlight
-- @
--
-- __BUG__: requestLatched is in arHandshake but not in masterOut.arvalid!
--
-- == State Transitions
--
-- @
--                 arHandshake
--     ┌────────┐ ───────────► ┌────────────┐
--     │  Idle  │              │  InFlight  │
--     │inFlight│              │ inFlight=T │
--     │ =False │ ◄─────────── │ owner=who  │
--     └────────┘ transDone    └────────────┘
-- @
--
-- == Output Signal Generation
--
-- === masterOut (to DRAM)
--
-- @
-- masterOut.arvalid = !inFlight && selectedArValid
--     -- BUG: Should include requestLatched!
--     -- Should be: !inFlight && (selectedArValid || requestLatched)
--
-- masterOut.ardata = masters[activeIdx].ardata
--     -- Forward selected master's address
--
-- masterOut.rready = masters[transactionOwner].rready
--     -- Forward owner's ready (only owner should be ready)
-- @
--
-- === perHeadSlaves[i] (to each head)
--
-- @
-- perHeadSlaves[i].arready = (activeIdx == i) && !inFlight && slaveIn.arready
--     -- Only active head sees arready, and only when idle
--
-- perHeadSlaves[i].rvalid = (transactionOwner == i) && inFlight && slaveIn.rvalid
--     -- Only owner sees rvalid
--
-- perHeadSlaves[i].rdata = slaveIn.rdata
--     -- Broadcast data (but non-owners ignore due to rvalid=False)
-- @
--
-- == Transaction Timeline
--
-- @
-- Cycle:        0    1    2    3    4    5    6    7    8
-- 
-- Head0.arvalid ─┐________________________________...
-- Head1.arvalid ───────┐____________________________...
-- 
-- inFlight:     F    F    T    T    T    T    F    F    T
-- activeIdx:    0    0    0    0    0    0    1    1    1
-- owner:        x    0    0    0    0    0    0    1    1
-- lastGranted:  x    x    x    x    x    0    0    0    0
-- 
-- masterOut.ar: ─────┐_____________┐____...
-- slaveIn.ardy: ─────┐_____________┐____...
-- slaveIn.rvalid: _______┐──┐__________┐...
-- slaveIn.rlast:  ___________┐__________...
-- 
-- perHead0.ardy: ────┐___________________...
-- perHead0.rvalid: ______┐──┐____________...
-- perHead1.ardy: _______________┐________...
-- perHead1.rvalid: __________________┐___...
-- @
--
-- == Critical Design Points
--
-- 1. __No AR while in-flight__: masterOut.arvalid is gated by !inFlight.
--    This prevents pipelining but ensures simple response routing.
--
-- 2. __Owner-based routing__: Responses go ONLY to transactionOwner.
--    Other heads see rvalid=False, preventing spurious data capture.
--
-- 3. __Round-robin fairness__: lastGranted updates on transaction COMPLETION,
--    not on grant. This ensures stuck transactions don't starve others.
--
-- 4. __Broadcast data__: rdata is broadcast to all heads (simpler routing),
--    but only owner's rvalid is True, so only owner latches it.
--
-- 5. __Request latching__: Prevents arvalid drops during multi-cycle waits,
--    but has implementation bug (not included in output arvalid).
--
-- == Why Response Routing Matters
--
-- Without proper routing, if Head 0 and Head 1 both request:
-- 1. Arbiter grants Head 1's request
-- 2. DRAM returns data for Head 1
-- 3. BUG: Head 0 sees rvalid (thinks its data arrived) and latches wrong data!
--
-- With proper routing:
-- 1. Arbiter grants Head 1, sets owner=1
-- 2. DRAM returns data
-- 3. perHeadSlaves[0].rvalid = False (owner != 0)
-- 4. perHeadSlaves[1].rvalid = True (owner == 1)
-- 5. Only Head 1 latches the data
--
-- == Known Issues
--
-- 1. requestLatched not included in masterOut.arvalid output
-- 2. Entire arbiter currently disabled in qkvProjector (qResults used instead of qResults')
--
-- == Usage Notes
--
-- 1. All masters must follow AXI protocol: hold arvalid until arready.
--
-- 2. Masters must be prepared to wait indefinitely for arready (arbitration).
--
-- 3. Masters should only assert rready when they can actually accept data.
--
-- 4. The arbiter assumes single-beat transactions (no burst support in 
--    current implementation, but rlast is checked for future extension).
--
-- 5. To enable this arbiter, modify qkvProjector to use qResults' instead of qResults.
--
axiArbiterWithRouting :: forall dom n.
  (HiddenClockResetEnable dom, KnownNat n)
  => Slave.AxiSlaveIn dom              -- ^ Single DRAM slave
  -> Vec n (Master.AxiMasterOut dom)   -- ^ Multiple masters (heads)
  -> ( Master.AxiMasterOut dom         -- ^ Combined master to DRAM
     , Vec n (Slave.AxiSlaveIn dom)    -- ^ Per-head slave interfaces
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
        nextRequestLatched = mux arHandshake (pure False)  -- Clear on handshake
                          $ mux selectedArValid (pure True)  -- Latch on request
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
      { arvalid = mux inFlight (pure False) selectedArValid  -- Don't issue AR while in-flight
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
