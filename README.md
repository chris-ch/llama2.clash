## llama2.clash

__WORK IN PROGRESS__

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

A **Haskell/Clash** implementation of the **Llama 2** decoder, inspired by Andrej Karpathy’s minimalist pure C reference [llama2.c](https://github.com/karpathy/llama2.c).

This repository started after I first ported `llama2.c` to plain Haskell in [llama2.hs](https://github.com/chris-ch/llama2.hs). That migration naturally led to the next question:

> *What would it look like to push the same architecture all the way into hardware description with Clash?*

`llama2.clash` is the result: an experiment in expressing a transformer decoder in idiomatic Haskell while keeping it synthesizable to Verilog via [Clash](https://clash-lang.org/).

Although the project is exploratory, it also has practical value as a **hardware/software co-design reference** for transformer inference. It provides a concrete way to study:

- how decoder components map to hardware,
- how DRAM bandwidth and on-chip memory affect performance,
- where latency bottlenecks appear as model sizes scale,
- and how a high-level typed description can still produce synthesizable RTL.

In that sense, this repository is best viewed as a **reference implementation for transformer hardware exploration**: useful for FPGA/ASIC architecture experiments, reproducible numeric validation, and as a starting point for custom inference accelerators.

At the moment, the repository supports:

- **software simulation** of several model sizes,
- **numeric validation** against Python/reference behavior,
- and **Verilog generation** for Clash-synthesizable top entities.

Notably, full Verilog generation is no longer limited to toy-scale models: in practice, even very large configurations are now tractable with Clash. On a regular laptop, I have successfully generated Verilog for:

- **Llama 2 13B** in **under 30 minutes**
- **Llama 2 70B** in **about 1 hour**

This makes the repository useful not just as a conceptual exercise, but as a realistic frontend for large-model hardware architecture experiments.

## Running llama2

You will need to download one of the pretrained model checkpoints, for example the TinyStories models from [Hugging Face](https://huggingface.co/karpathy/tinyllamas/tree/main):

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Larger checkpoints are also available for better output quality:

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

Once a model is downloaded, you can run `llama2` immediately:

```shell
cabal run llama2 --flag model-15m -- --temperature 0.8 --seed 123 "In that little town"
```

Example output (here using the 110M model):

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

| Command          | Model             | Time    | When to use                    |
| ---------------- | ----------------- | ------- | ------------------------------ |
| `make test`      | nano (ModelDim=8) | ~2-3 min   | Day-to-day development         |
| `make test-full` | 260K              | ~10 min | Pre-merge / numeric validation |

```shell
# Fast tests (nano model — all unit + integration tests)
make test

# Full numeric validation (260K model — checks exact layer norms vs Python reference)
make test-full
```

The nano model has tiny dimensions (`ModelDimension=8`, `HeadDimension=2`, 2 layers), so every DRAM fetch is a single beat and simulation finishes quickly.

To run a single test by name:

```shell
make test ARGS='--test-options="--match \"Layer 0 output norm\""'
# or directly:
cabal test llama2-test -f model-nano -f -model-260k \
  --test-show-details=direct \
  --test-options='--match "Layer 0 output norm"'
```

## Verilog generation

`clash-ghc` is declared as a `build-tool-depends` in `llama2.cabal`, so Cabal fetches and builds the correct version automatically — no global install is needed.

Use `synth.sh` to generate Verilog:

```shell
# Full synthesis (default)
./synth.sh model-nano
./synth.sh model-7b
./synth.sh model-13b
./synth.sh model-70b

# Hierarchical synthesis, bottom-up (optional — useful for isolating OOM failures)
# Each block runs in its own Clash process; stops at the first failure.
./synth.sh model-70b hierarchical
```

All model variants from `ModelConfig.hs` are supported:

| Flag         | dim  | heads (Q/KV) | layers | VocabSize | Notes                                   |
| ------------ | ---- | ------------ | ------ | --------- | --------------------------------------- |
| `model-nano` | 8    | 4/2          | 2      | 512       | Tiny dims, fast simulation tests        |
| `model-260k` | 64   | 8/4          | 5      | 512       | Default; 260K parameter reference model |
| `model-15m`  | 288  | 6/6          | 6      | 32000     | Requires `./data/stories15M.bin`        |
| `model-42m`  | 512  | 8/8          | 8      | 32000     | Requires `./data/stories42M.bin`        |
| `model-110m` | 768  | 12/12        | 12     | 32000     | Requires `./data/stories110M.bin`       |
| `model-7b`   | 4096 | 32/32        | 32     | 32000     | Synthesis only                          |
| `model-13b`  | 5120 | 40/40        | 40     | 32000     | Synthesis only                          |
| `model-70b`  | 7168 | 64/64        | 70     | 32000     | Synthesis only                          |

Output Verilog is written to `/tmp/clash-verilog-<model>/`, with one subdirectory per block. Per-block logs (`clash-1-sampler.log`, etc.) are kept there for inspection.

Hierarchical mode synthesises five blocks independently, bottom-up:

| Step | Block             | Entry point                           |
| ---- | ----------------- | ------------------------------------- |
| 1    | Token sampler     | `Sampler.samplerTop`                  |
| 2    | Input embedding   | `InputEmbedding.inputEmbeddingTop`    |
| 3    | Output projection | `OutputProjection.logitsProjectorTop` |
| 4    | Layer runner      | `LayerRunner.layerRunnerTop`          |
| 5    | Full decoder      | `Decoder.topEntity`                   |

### Verified synthesis scale

Full Verilog generation has been successfully validated for all current synthesis-oriented configurations, including large Llama 2 class shapes:

* **`model-7b`** — validated
* **`model-13b`** — validated (**< 30 min** on a regular laptop)
* **`model-70b`** — validated (**~1 hour** on a regular laptop)

This is a useful data point for Clash users: even extremely large statically-typed transformer descriptions can still elaborate and normalize into synthesizable RTL in practical time.

### Memory and runtime notes

Clash normalization is RAM-intensive for large models, and runtime scales significantly with model size. However, the current structure is now robust enough that even very large variants can complete on commodity hardware.

* `model-nano` synthesises comfortably on a 32 GB machine.
* `model-7b`, `model-13b`, and `model-70b` have all been validated through full Verilog generation.
* If a full synthesis run fails on your machine, use **hierarchical mode** to isolate the bottleneck and reduce per-process memory pressure.

**Important:** “Verilog generation succeeded” means the Clash frontend completed elaboration/normalization and emitted RTL. It does **not** imply that downstream FPGA implementation is easy or practical at those scales without substantial architecture-specific optimization.

**Downstream EDA is out of scope for this repository.** FPGA implementation (LUT/FF/BRAM resource counts, timing closure, place-and-route) requires running the generated Verilog through a vendor tool such as Vivado (Xilinx/AMD) or Quartus (Intel/Altera). The design targets 400 MHz with an estimated ~512 DSP blocks, as noted in `PRODUCT.md`.

## FPGA mapping — Vivado exploratory results

### Target device

The part list available under the free Vivado WebPACK edition does not include the KV260's
exact `xck26-sfvc784-2LV-c` part. After reviewing all available parts, the recommended
target is **`xczu7ev-ffvc1156-3-e`** — the highest-tier device in the list and the best
fit for this design.

| Feature | Original target (ZU9EG) | Recommended (ZU7EV-3-E) |
| --- | --- | --- |
| DSPs | 2,520 | 1,728 |
| LUTs | ~274k | 230,400 |
| Speed grade | -1 / -2 | **-3** (highest available) |
| UltraRAM | varies | **96 blocks (32.4 Mb)** |
| Processors | Quad-core A53 | Quad-core A53 |

The ZU9EG has more DSPs and logic, but it is not in the WebPACK part list at all. The
ZU7EV-3-E compensates with the highest speed grade (critical for AXI streaming and
row-wise burst throughput) and 96 UltraRAM blocks, which are well-suited for KV cache
storage. The ZU5EG (`xczu5eg-sfvc784-2-e`) is the closest K26/KV260 proxy but offers
fewer resources and a lower speed grade — it is outclassed by the ZU7EV for this workload.


## Migrating to LLaMa 3

LLaMa 3 shares the same decoder-only transformer skeleton, so the RTL blocks (`Decoder`,
`LayerRunner`, `MultiHeadAttention`, `FeedForwardNetwork`, AXI memory subsystem) carry
over unchanged. This section describes a **clean cut-over** — LLaMa 2 support is dropped
entirely. No CPP guards are kept for the old model; files are updated in place.

### 1. Model configuration (`LLaMa2/Types/ModelConfig.hs`)

Delete all existing `#elif MODEL_*` branches and replace the two synthesis-target
configurations with:

| Model       | dim  | hidden | layers | Q heads | KV heads | head dim | vocab   | seq  |
| ----------- | ---- | ------ | ------ | ------- | -------- | -------- | ------- | ---- |
| LLaMa 3 8B  | 4096 | 14336  | 32     | 32      | 8        | 128      | 128,256 | 8192 |
| LLaMa 3 70B | 8192 | 28672  | 80     | 64      | 8        | 128      | 128,256 | 8192 |

`NumKeyValueHeads` already exists as a type parameter, so GQA is structurally supported.
The jump to `VocabularySize = 128256` is the most impactful change: it widens both the
embedding table and the output-projection matrix, and it directly drives the `ram_init`
overflow issue described above — making the `blockRamU` fix a prerequisite for synthesis.
Verify that `BankDepth` and `BankAddress` still fit in your chosen index type under the
new `SequenceLength = 8192`.

### 2. Rotary positional encoding (`LLaMa2/Layer/Attention/RotaryEncoding.hs`)

Update the RoPE base frequency in place: **10,000 → 500,000**. Recompute the
`freqCosF` / `freqSinF` tables baked into `RotaryEncodingComponentF` with the new theta.
No structural change to `rotaryPositionEncoder` is required.

### 3. Tokenizer (`Tokenizer.hs`)

Replace `buildTokenizer` entirely. The LLaMa 2 SentencePiece binary parser is gone;
LLaMa 3 uses a **tiktoken BPE** tokenizer (`tokenizer.model`):

- Write a new tiktoken-format reader: base-64 encoded merge rules + a rank table.
- Replace `encodeTokens` with a BPE merge loop on the new rank table (drop the
  SentencePiece score-based merge logic).
- Update `decodePiece` byte-fallback to tiktoken's byte-level vocabulary encoding
  (`<0xNN>` handling changes slightly).

The hardware path (`InputEmbedding`, `OutputProjection`) is unaffected.

### 4. Checkpoint / weight parser (`Parser.hs`)

Replace the raw-binary checkpoint parser entirely. LLaMa 3 weights ship as **safetensors**:

- Read the safetensors JSON header to extract tensor names, dtypes, and byte offsets.
- Map Meta's canonical tensor names (`model.layers.N.self_attn.q_proj.weight`, etc.) to
  the flat weight layout expected by `WeightsLayout.hs`.
- LLaMa 3 weights are **bfloat16**; add a bfloat16 → float32 conversion step at the
  parser boundary before handing values to `ParametersQuantization`.

### 5. Weights memory layout (`LLaMa2/Memory/WeightsLayout.hs`, `KVCacheLayout.hs`)

LLaMa 3 drops to 8 KV heads for both the 8B and 70B variants, significantly shrinking K
and V weight matrices relative to Q. Update the DRAM address arithmetic in `WeightsLayout`
for the new `NumQueryHeads / NumKeyValueHeads` ratio and confirm regions remain
non-overlapping. Recalculate `KVCacheLayout`'s `BankDepth` for `SequenceLength = 8192`.

### 6. Cabal flags (`llama2.cabal`)

Remove all `model-7b`, `model-13b`, `model-70b` (and smaller LLaMa 2) flags. Replace with
`model-llama3-8b` and `model-llama3-70b`. Keep `model-nano` and `model-260k` as fast
simulation targets if desired, but update their CPP guards to be LLaMa-3-neutral names.

### Summary of files to change

| File | Change |
| ---- | ------ |
| `LLaMa2/Types/ModelConfig.hs` | Replace all `MODEL_*` branches with `MODEL_LLAMA3_8B` / `MODEL_LLAMA3_70B` |
| `LLaMa2/Layer/Attention/RotaryEncoding.hs` | Update RoPE theta base (10,000 → 500,000) |
| `Tokenizer.hs` | Replace SentencePiece parser with tiktoken BPE reader |
| `Parser.hs` | Replace binary checkpoint parser with safetensors reader + bfloat16 conversion |
| `LLaMa2/Memory/WeightsLayout.hs` | Update address map for new GQA ratio and vocab size |
| `LLaMa2/Memory/KVCacheLayout.hs` | Recalculate bank sizing for `SequenceLength = 8192` |
| `llama2.cabal` | Replace LLaMa 2 flags with `model-llama3-8b` / `model-llama3-70b` |

---

## Simulation timing (260K model, all DRAM-backed)

Measured on `--temperature 0 --seed 123 "Hi"` with the 260K model (dim=64, 5 layers, 8Q/4KV heads, vocab=512).

### Stage durations per layer

| Stage | Cycles | Notes |
| ----- | ------ | ----- |
| QKV projection | ~792–801 | slightly longer for layer 0 (embedding fetch on first token) |
| Multi-head attention + WO | ~1569–1591 | grows by ~11 cycles per token as KV sequence lengthens |
| FFN (RMS norm + W1/W3/W2) | ~4900 | constant — no sequence-length dependence |
| **Per-layer total** | **~7261–7280** | |
| Output projection + sampling | ~5777 | constant |
| **Per-token total** | **~42,095–42,203** | grows slowly with sequence position |

### Per-token cycle events (prompt `[1, 320, 417]` → `" H"`, `"i"`, `"b"`)

**Token 0 (BOS, seq pos 0)**

| Layer | Start | QKVDone | AttnDone | FFNDone | ffnOutput[0] |
| ----- | ----- | ------- | -------- | ------- | ------------ |
| 0     | 0     | 801     | 2370     | 7270    | 0.3445       |
| 1     | 7271  | 8063    | 9632     | 14532   | 0.7467       |
| 2     | 14533 | 15325   | 16894    | 21794   | 0.6990       |
| 3     | 21795 | 22587   | 24156    | 29056   | 0.6362       |
| 4     | 29057 | 29849   | 31418    | 36318   | 0.9740       |

Ready pulse at cycle 42095. Output token: `" H"`.

**Token 1 (" H", seq pos 1)**

| Layer | Start | QKVDone | AttnDone | FFNDone | ffnOutput[0] |
| ----- | ----- | ------- | -------- | ------- | ------------ |
| 0     | 42096 | 42895   | 44475    | 49375   | −0.1266      |
| 1     | 49376 | 50168   | 51748    | 56648   | −0.1755      |
| 2     | 56649 | 57441   | 59021    | 63921   | 0.1231       |
| 3     | 63922 | 64714   | 66294    | 71194   | 0.7042       |
| 4     | 71195 | 71987   | 73567    | 78467   | 1.7086       |

Ready pulse at cycle 84244. Output token: `"i"`.

**Token 2 ("i", seq pos 2)**

| Layer | Start  | QKVDone | AttnDone | FFNDone | ffnOutput[0] |
| ----- | ------ | ------- | -------- | ------- | ------------ |
| 0     | 84245  | 85044   | 86635    | 91535   | 0.4646       |
| 1     | 91536  | 92328   | 93919    | 98819   | 0.8395       |
| 2     | 98820  | 99612   | 101203   | 106103  | 1.1952       |
| 3     | 106104 | 106896  | 108487   | 113387  | 1.2483       |
| 4     | 113388 | 114180  | 115771   | 120671  | 1.3673       |

Ready pulse at cycle 126448. Output token: `"b"`.

Generated tokens: `" H"`, `"i"`, `"b"` … (matching the gold-standard `" Hibo and Anna…"` output).
