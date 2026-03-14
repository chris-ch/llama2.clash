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

# Verilog / VHDL generation

`clash-ghc` is declared as a `build-tool-depends` in `llama2.cabal`, so cabal
fetches and builds the correct version automatically — no global install needed.

Use `synth.sh` to generate Verilog for either model configuration:

```shell
# 260K model (default)
./synth.sh model-260k

# Tiny NANO model (fast, for smoke-testing the flow)
./synth.sh model-nano
```

Output Verilog files are written to `/tmp/clash-verilog-<model>/` and a
`clash.log` is kept there for inspection.

Or invoke Clash directly through cabal:

```shell
cabal exec -- clash --verilog -DMODEL_260K \
  -iproject/llama2 \
  -outputdir /tmp/clash-build \
  project/llama2/LLaMa2/Decoder/Decoder.hs
```

**Memory note:** HDL elaboration of `topEntity` is RAM-intensive because
`Vec VocabularySize FixedPoint` (512 elements) forces Clash to normalise a
large comparison tree in-memory. The NANO model used ~21 GB before OOM on a
30 GB machine; the 260K model did not finish in 10+ hours. A machine with
≥ 64 GB RAM is recommended. FPGA resource counts (LUT/FF/BRAM) require Vivado
or Quartus on the generated Verilog.


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
