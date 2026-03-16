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
| `make test`      | nano (ModelDim=8) | ~18 s   | Day-to-day development         |
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

# Hierarchical synthesis, bottom-up (optional — useful for isolating OOM failures)
# Each block runs in its own Clash process; stops at the first failure.
./synth.sh model-7b hierarchical
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

**Memory:** Clash normalization is RAM-intensive for large models. `model-nano` synthesises comfortably on a 32 GB machine. All variants up to and including `model-7b` have been validated through full Verilog generation. Use hierarchical mode if a full synthesis run runs out of memory — it helps isolate which block is the bottleneck.

**Downstream EDA is out of scope for this repository.** FPGA implementation (LUT/FF/BRAM resource counts, timing closure, place-and-route) requires running the generated Verilog through a vendor tool such as Vivado (Xilinx/AMD) or Quartus (Intel/Altera). The design targets 400 MHz with an estimated ~512 DSP blocks, as noted in `PRODUCT.md`.

## Simulation timing (260K model, all DRAM-backed)

With all weight matrices fetched from DRAM via AXI, each transformer layer takes approximately 7,000 simulation cycles. A complete token (5 layers + classifier) takes roughly 38,000 cycles.

Layer-level norm reference values for `--temperature 0 --seed 123 "Hi"` (token 0, BOS):

| Layer | norm(attn) | norm(output) |
| ----- | ---------- | ------------ |
| 0     | 3.0385     | 8.2357       |
| 1     | 9.1765     | 10.3825      |
| 2     | 11.2580    | 12.2522      |
| 3     | 14.7369    | 16.0392      |
| 4     | 16.4430    | 18.7545      |

Generated tokens: `" H"`, `"i"`, `"b"` … (matching the gold-standard `" Hibo and Anna…"` output).
