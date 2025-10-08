# Row State Machine Specification

## Purpose
Manages sequential row-by-row processing of a matrix-vector multiplication, coordinating when to process each row and signaling when the row processor should be reset.

## States

### IDLE
- **Description**: No processing is active. Ready to accept new work.
- **Outputs**: 
  - `busy = False`
  - `rowIdx = 0`
  - `clearRow = True` (on entry, including initial cycle 0)

### PROCESSING(n)
- **Description**: Currently processing row `n` where `0 <= n < numRows`
- **Outputs**:
  - `busy = True`
  - `rowIdx = n`
  - `clearRow = True` (on entry only, single cycle pulse)

## State Transitions

```
         ┌─────────────────────────────────────┐
         │                                     │
         ▼                                     │
    ┌────────┐                                 │
    │  IDLE  │                                 │
    └────────┘                                 │
         │                                     │
         │ validIn=True                        │
         │                                     │
         ▼                                     │
┌──────────────────┐                          │
│  PROCESSING(0)   │                          │
└──────────────────┘                          │
         │                                     │
         │ rowDone=True                        │
         │                                     │
         ▼                                     │
┌──────────────────┐                          │
│  PROCESSING(1)   │                          │
└──────────────────┘                          │
         │                                     │
         │ rowDone=True                        │
         │                                     │
         ▼                                     │
        ...                                    │
         │                                     │
         │ rowDone=True                        │
         │                                     │
         ▼                                     │
┌──────────────────┐                          │
│ PROCESSING(N-1)  │ ← last row               │
└──────────────────┘                          │
         │                                     │
         │ rowDone=True                        │
         └─────────────────────────────────────┘
```

## Transition Rules

### From IDLE
- **Condition**: `validIn = True`
- **Action**: Transition to `PROCESSING(0)`
- **Outputs**: `busy = True`, `rowIdx = 0`, `clearRow = True`
- **Notes**: Only responds to `validIn` when idle

### From PROCESSING(n) where n < maxRow
- **Condition**: `rowDone = True`
- **Action**: Transition to `PROCESSING(n+1)`
- **Outputs**: `busy = True`, `rowIdx = n+1`, `clearRow = True`
- **Notes**: Advance to next row

### From PROCESSING(maxRow)
- **Condition**: `rowDone = True`
- **Action**: Transition to `IDLE`
- **Outputs**: `busy = False`, `rowIdx = 0`, `clearRow = True`
- **Notes**: All rows complete, return to idle

### Hold in current state
- **Condition**: No transition conditions met
- **Action**: Remain in current state
- **Outputs**: Same as current state, but `clearRow = False`

## Signal Semantics

### Inputs
- **`validIn`**: Request to start processing. Only honored when in IDLE state.
- **`rowDone`**: Current row has completed processing. Triggers advancement to next row or return to IDLE.

### Outputs
- **`busy`**: High when processing any row, low when idle.
- **`rowIdx`**: Index of the row currently being processed (0 when idle).
- **`clearRow`**: Single-cycle pulse that goes high when entering ANY state (including the initial IDLE state at reset). Signals downstream logic to reset/initialize for the current row.

## Critical Timing Properties

1. **Initial Reset**: On the very first cycle (cycle 0), the state machine outputs `IDLE` with `clearRow = True`. This ensures all downstream logic starts in a known state.

2. **Entry Pulses**: `clearRow` pulses high for exactly ONE cycle when entering any state:
   - Cycle 0: Initial IDLE → `clearRow = True`
   - When validIn triggers: IDLE → PROCESSING(0) → `clearRow = True`
   - When rowDone advances: PROCESSING(n) → PROCESSING(n+1) → `clearRow = True`
   - When completing: PROCESSING(maxRow) → IDLE → `clearRow = True`

3. **Stable State**: When holding in a state (no transition), `clearRow = False`.

4. **No Spurious Transitions**: 
   - `validIn` is ignored when `busy = True`
   - `rowDone` only has effect when in a PROCESSING state
   - Multiple consecutive `rowDone` pulses only advance one row per pulse

5. **Synchronous Operation**: All outputs change synchronously with the clock edge. The `clearRow` pulse is visible on the same cycle as the new `rowIdx`.

## Example Sequence

For a 3-row matrix with `rowDone` pulsing at appropriate times:

| Cycle | validIn | rowDone | State          | busy | rowIdx | clearRow | Notes |
|-------|---------|---------|----------------|------|--------|----------|-------|
| 0     | 0       | 0       | IDLE           | 0    | 0      | 1        | Initial reset |
| 1     | 1       | 0       | PROCESSING(0)  | 1    | 0      | 1        | Start row 0 |
| 2     | 0       | 0       | PROCESSING(0)  | 1    | 0      | 0        | Working on row 0 |
| 3     | 0       | 1       | PROCESSING(1)  | 1    | 1      | 1        | Row 0 done, start row 1 |
| 4     | 0       | 0       | PROCESSING(1)  | 1    | 1      | 0        | Working on row 1 |
| 5     | 0       | 1       | PROCESSING(2)  | 1    | 2      | 1        | Row 1 done, start row 2 |
| 6     | 0       | 0       | PROCESSING(2)  | 1    | 2      | 0        | Working on row 2 |
| 7     | 0       | 1       | IDLE           | 0    | 0      | 1        | Row 2 done, return to idle |
| 8     | 0       | 0       | IDLE           | 0    | 0      | 0        | Stable idle |

## Design Rationale

### Why pulse clearRow on EVERY state entry?
This creates a uniform "entry action" semantic. Downstream logic can rely on seeing a clear pulse whenever the state changes, simplifying integration. The alternative (only clearing when "needed") requires downstream logic to track state changes themselves.

### Why clear on initial cycle?
Defensive programming. Even though we're idle, pulsing clear ensures any connected logic initializes properly. Without this, there could be ambiguity about the initial state of downstream processors.

### Why ignore validIn when busy?
This prevents race conditions and simplifies the protocol. The caller must wait for `busy` to go low before issuing a new `validIn`. This creates clear transaction boundaries.

### Why is rowIdx = 0 when idle?
Consistent default value. Since we're not processing, the actual value doesn't matter, but 0 is the natural "no row selected" indicator.
