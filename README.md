## llama2.clash

__WORK IN PROGRESS__

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

An implementation in *Haskell*/[*Clash*](https://clash-lang.org/) based on Andrej Karpathy's [Llama 2](https://ai.meta.com/llama/) model in pure C [llama2.c](https://github.com/karpathy/llama2.c).

## Running the llama2

You will need to install a few training sets,
for example the mini stories from [Hugging Face](https://huggingface.co/karpathy/tinyllamas/tree/main):

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

There are also bigger models, for better stories:

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

Once a model is downloaded, you can then run the llama2 right away: 

```shell
cabal run llama2 --flag model-15m -- --temperature 0.8 --seed 123 "In that little town"
```

This is the kind of output you will get (here using the 110M model):

```text
<s>
In that little town, there was a humble house. In the house lived a kind man named Tom. Tom had a big potato farm. He loved to grow potatoes and share them with his friends.
One day, a little girl named Lily came to Tom's house. She was hungry and asked, "Can I have a potato, please?" Tom smiled and said, "Of course, Lily! I have many potatoes to offer you."
Tom gave Lily a big potato from his farm. Lily was very happy and said, "Thank you, Tom!" She went back to her home and ate the potato. It was the best potato she had ever tasted.
The next day, Lily came back to Tom's house with a big smile. She had a big bag of coins. "Tom, I want to give you this coins to say thank you for the potato," she said. Tom was very happy and thanked Lily for the coins.
From that day on, Lily and Tom became good friends. They would often talk and share potatoes from the humble little house. And they all lived happily ever after.
<s>
```

## Reproducible output

```shell
cabal run llama2 --flag model-15m -- --temperature 0 --seed 123 "Hi"
```

```text
 Hippy was a very happy dog. He loved to play in the park with his friends. One day, he was playing with a ball and he was running around with his friends. Suddenly, he saw a big, scary dog. He was so scared that he started to bark and growl.
Hippy's friends were scared too and they ran away. [...]"
```

For debugging / testing with the 260K model:

```shell
cabal run llama2 --flag model-260k -- --temperature 0 --seed 123 "Hi"
```

```text
<s>
 Hibo and Anna, Anna, Anna, came to visit her. She saw a big box and asked Anna. [...]
```

## Testing

The test suite supports two model configurations:

| Command | Model | Time | When to use |
|---------|-------|------|-------------|
| `make test` | nano (ModelDim=8) | ~18 s | Day-to-day development |
| `make test-full` | 260K | ~10 min | Pre-merge / numeric validation |

```shell
# Fast tests (nano model — all unit + integration tests)
make test

# Full numeric validation (260K model — checks exact layer norms vs Python reference)
make test-full
```

The nano model has tiny dimensions (ModelDimension=8, HeadDimension=2, 2 layers) so
every DRAM fetch is a single beat and simulation finishes quickly.

To run a single test by name:

```shell
make test ARGS='--test-options="--match \"Layer 0 output norm\""'
# or directly:
cabal test llama2-test -f model-nano -f -model-260k \
  --test-show-details=direct \
  --test-options='--match "Layer 0 output norm"'
```

## C Version

```shell
haskell@a050ba3ea910:/workspaces/llama2.clash$ /usr/bin/time -v ./run data/stories110M.bin -t 0.8 -n 256 -s 123 -i "In that little town"
achieved tok/s: 15.105312
        Command being timed: "./run data/stories110M.bin -t 0.8 -n 256 -i In that little town"
        User time (seconds): 14.15
        System time (seconds): 0.05
        Percent of CPU this job got: 99%
        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:14.21
        Average shared text size (kbytes): 0
        Average unshared data size (kbytes): 0
        Average stack size (kbytes): 0
        Average total size (kbytes): 0
        Maximum resident set size (kbytes): 447516
        Average resident set size (kbytes): 0
        Major (requiring I/O) page faults: 0
        Minor (reclaiming a frame) page faults: 11291
        Voluntary context switches: 1
        Involuntary context switches: 40
        Swaps: 0
        File system inputs: 0
        File system outputs: 0
        Socket messages sent: 0
        Socket messages received: 0
        Signals delivered: 0
        Page size (bytes): 4096
        Exit status: 0
```
# Verilog / VHDL generation

Install:

```shell
cabal install clash-ghc --overwrite-policy=always
```

Run:

```shell
cabal exec -- clash --verilog \
  -iproject/llama2 \
  -package llama2 \
  -outputdir /tmp/clash-build \
  -DMODEL_260K \
  -fclash-inline-limit=20 \
  -fclash-spec-limit=10 \
  -fclash-clear \
  project/llama2/LLaMa2/Decoder/Decoder.hs
```

## Handshaking conventions

| Signal        | Direction | Purpose                        | Behavior                                                                                                 |
| ------------- | --------- | ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **Ready Out** | Output    | Indicates consumer readiness   | Asserted (high) when the consumer can accept/process data; deasserted when busy (e.g., stalled or full). |
| **Valid Out** | Output    | Indicates valid output data    | Asserted (high, level) when new data is available; held high until the transaction completes (i.e., until `readyIn` is also high). |
| **Ready In**  | Input     | Indicates downstream readiness | Producer transfers data only when `readyIn` is high; holds `validOut` and data stable if `readyIn` is low. |
| **Valid In**  | Input     | Indicates valid input data     | Consumer processes `dataIn` only when `validIn` is high and it asserts `readyOut`.                       |

## SUMMARY OF TOKEN FLOW
1. Pipeline controller enters Stage1 for layer L
2. isStage1ThisLayer goes HIGH (inValid to QKV controller)
3. QKV controller transitions: Idle -> Computing
4. computeEnable stays HIGH, feeding all matrix multipliers
5. All heads compute in parallel (multiple cycles per matvec)
6. When ALL heads complete, matVecValid goes HIGH
7. QKV controller transitions: Computing -> Done (outValid HIGH)
8. Pipeline controller sees qkvValidThisLayer, advances to Stage2
9. Pipeline advances, isStage1ThisLayer goes LOW (outReady asserted)
10. QKV controller transitions: Done -> Idle


# Key Design Decisions Summary

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Weight Storage** | eMMC 5.1 | Non-volatile, reprogrammable, high-density embedded storage |
| **Weight Format** | I8E (8-bit + 7-bit exp) | 4× compression vs FixedPoint, minimal quality loss |
| **Compute Format** | FixedPoint (SFixed 12 20) | Native FPGA arithmetic, good precision |
| **Working Memory** | DDR4 DRAM | Fast random access for KV cache, affordable |
| **Layer Strategy** | One at a time + prefetch | Fits in available memory, hides ROM latency |
| **KV Cache** | External DDR4 | Too large for on-chip, needs fast random access |

# Simulation timing (260K model, all DRAM-backed)

With all weight matrices fetched from DRAM via AXI, each transformer layer takes approximately
7,000 simulation cycles. A complete token (5 layers + classifier) takes roughly 38,000 cycles.

Layer-level norm reference values for `--temperature 0 --seed 123 "Hi"` (token 0, BOS):

| Layer | norm(attn) | norm(output) |
|-------|-----------|--------------|
| 0     | 3.0385    | 8.2357       |
| 1     | 9.1765    | 10.3825      |
| 2     | 11.2580   | 12.2522      |
| 3     | 14.7369   | 16.0392      |
| 4     | 16.4430   | 18.7545      |

Generated tokens: " H", "i", "b" … (matching gold standard " Hibo and Anna…" output).
