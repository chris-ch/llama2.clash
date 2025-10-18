Why we sometimes don't use ready signals: they aren't needed in the current architecture.

The key insight is that the architecture uses **transaction-based control** rather than **streaming backpressure**:

- **One token processes completely** through all stages before the next token enters
- The **FSM controls when computations start and finish** - it doesn't ask the datapath "are you ready?", it just starts it and waits for completion
- **No overlap** between stages for the same layer, so no resource contention

The `readyIn` signals would only be needed if you wanted:
1. Multiple tokens in flight simultaneously (pipelined)
2. Shared resources that could be busy
3. Downstream buffers that could overflow

Since we have a single-token pipeline where each stage completes before the next starts, the FSM-level coordination (`qkvOutReady`, `matVecValid`) is sufficient.

The signals are present in the interfaces because they follow a standard ready/valid protocol pattern, which makes the code more composable and easier to extend later if you want to add pipelining. But for now, they can safely be tied to `(pure True)` or left unused.
