# To do

**Implementation Phases (AXI & weight path)**

1. AXI primitives (types, single-beat read, burst read ≤256, optional minimal write)
2. Throughput & robustness (burst rules, error handling, measurement)
3. Weight loading & format (boot-time FP32→I8E converter + persist, layer prefetch)
4. Decoder integration (remove const weights, add AXI row fetchers, layer FSM)


