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

## Running a single test

```shell
cabal test --test-show-details=direct --test-options='--match "Layer 0 output norm"'
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
  project/llama2/LLaMa2/Top.hs
```

## Handshaking conventions

| Signal        | Direction | Purpose                        | Behavior                                                                                                 |
| ------------- | --------- | ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **Ready Out** | Output    | Indicates consumer readiness   | Asserted (high) when the consumer can accept/process data; deasserted when busy (e.g., stalled or full). |
| **Valid Out** | Output    | Indicates valid output data    | Asserted (high) for one clock cycle (pulse) when new data is available; deasserted otherwise.            |
| **Ready In**  | Input     | Indicates downstream readiness | Producer transfers data (keeps `validOut` high) only when `readyIn` is high; holds data if low.          |
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
| **Weight Storage** | NVMe SSD (ROM) | Non-volatile, reprogrammable, 10× cheaper than RAM |
| **Weight Format** | I8E (8-bit + 7-bit exp) | 4× compression vs FixedPoint, minimal quality loss |
| **Compute Format** | FixedPoint (SFixed 12 20) | Native FPGA arithmetic, good precision |
| **Working Memory** | DDR4 DRAM | Fast random access for KV cache, affordable |
| **Layer Strategy** | One at a time + prefetch | Fits in available memory, hides ROM latency |
| **KV Cache** | External DDR4 | Too large for on-chip, needs fast random access |

# Few first cycles

```text
Prepending BOS to prompt tokens: [1,320,417]
✅ model loaded successfully
✅ Prompt: [1,320,417]
Simulating 259000 cycles...
This may take a moment...

Loading: ./data/stories260K.bin
✅ Simulation Model (FPGA wired) Loaded

Cycle | Layer | DataStage      | Tok Rdy | QKVDone | AttnDone | FFNDone | WgtValid | LayerChg | norm(attn) | norm(out) |   Tok    |  SysState   | LayerValid
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    0 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |    False |     0.0000 |    0.0000 | "\n<s>\n" |     WSReady |       True
    1 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |    False |     0.0000 |    0.0000 | "\n<s>\n" |     WSReady |       True
   27 |     0 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     0.0000 |    0.0000 | "\n<s>\n" | WSStreaming |      False
  226 |     0 | ProcessingLayer |   False |   False |     True |   False |    False |    False |     3.0385 |    0.0000 | "\n<s>\n" | WSStreaming |      False
 1587 |     0 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     3.0385 |    8.2357 | "\n<s>\n" |     WSReady |      False
 1588 |     1 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     3.0385 |    8.2357 | "\n<s>\n" |     WSReady |       True
 1614 |     1 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     3.0385 |    8.2357 | "\n<s>\n" | WSStreaming |      False
 1813 |     1 | ProcessingLayer |   False |   False |     True |   False |    False |    False |     9.1765 |    8.2357 | "\n<s>\n" | WSStreaming |      False
 3174 |     1 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     9.1765 |   10.3825 | "\n<s>\n" |     WSReady |      False
 3175 |     2 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     9.1765 |   10.3825 | "\n<s>\n" |     WSReady |       True
 3201 |     2 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     9.1765 |   10.3825 | "\n<s>\n" | WSStreaming |      False
 3400 |     2 | ProcessingLayer |   False |   False |     True |   False |    False |    False |    11.2580 |   10.3825 | "\n<s>\n" | WSStreaming |      False
 4761 |     2 | ProcessingLayer |   False |   False |    False |    True |    False |    False |    11.2580 |   12.2522 | "\n<s>\n" |     WSReady |      False
 4762 |     3 | ProcessingLayer |   False |   False |    False |   False |    False |     True |    11.2580 |   12.2522 | "\n<s>\n" |     WSReady |       True
 4788 |     3 | ProcessingLayer |   False |    True |    False |   False |     True |    False |    11.2580 |   12.2522 | "\n<s>\n" | WSStreaming |      False
 4987 |     3 | ProcessingLayer |   False |   False |     True |   False |    False |    False |    14.7369 |   12.2522 | "\n<s>\n" | WSStreaming |      False
 6348 |     3 | ProcessingLayer |   False |   False |    False |    True |    False |    False |    14.7369 |   16.0392 | "\n<s>\n" |     WSReady |      False
 6349 |     4 | ProcessingLayer |   False |   False |    False |   False |    False |     True |    14.7369 |   16.0392 | "\n<s>\n" |     WSReady |       True
 6375 |     4 | ProcessingLayer |   False |    True |    False |   False |     True |    False |    14.7369 |   16.0392 | "\n<s>\n" | WSStreaming |      False
 6574 |     4 | ProcessingLayer |   False |   False |     True |   False |    False |    False |    16.4430 |   16.0392 | "\n<s>\n" | WSStreaming |      False
 7935 |     4 | ProcessingLayer |   False |   False |    False |    True |    False |    False |    16.4430 |   18.7545 | "\n<s>\n" |     WSReady |      False
 9472 |     4 | Classifier     |    True |   False |    False |   False |    False |    False |    16.4430 |   18.7545 | "\n<s>\n" |     WSReady |      False
 9473 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |     True |    16.4430 |   18.7545 |     " H" |     WSReady |       True
 9499 |     0 | ProcessingLayer |   False |    True |    False |   False |     True |    False |    16.4430 |   18.7545 |     " H" | WSStreaming |      False
 9699 |     0 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     2.7743 |   18.7545 |     " H" | WSStreaming |      False
10000 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |    False |     2.7743 |   18.7545 |     " H" | WSStreaming |      False
11060 |     0 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     2.7743 |    3.8518 |     " H" |     WSReady |      False
11061 |     1 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     2.7743 |    3.8518 |     " H" |     WSReady |       True
11087 |     1 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     2.7743 |    3.8518 |     " H" | WSStreaming |      False
11287 |     1 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     4.5273 |    3.8518 |     " H" | WSStreaming |      False
12648 |     1 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     4.5273 |    5.3427 |     " H" |     WSReady |      False
12649 |     2 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     4.5273 |    5.3427 |     " H" |     WSReady |       True
12675 |     2 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     4.5273 |    5.3427 |     " H" | WSStreaming |      False
12875 |     2 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     6.4987 |    5.3427 |     " H" | WSStreaming |      False
14236 |     2 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     6.4987 |    6.8089 |     " H" |     WSReady |      False
14237 |     3 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     6.4987 |    6.8089 |     " H" |     WSReady |       True
14263 |     3 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     6.4987 |    6.8089 |     " H" | WSStreaming |      False
14463 |     3 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     8.5103 |    6.8089 |     " H" | WSStreaming |      False
15824 |     3 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     8.5103 |    9.4203 |     " H" |     WSReady |      False
15825 |     4 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     8.5103 |    9.4203 |     " H" |     WSReady |       True
15851 |     4 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     8.5103 |    9.4203 |     " H" | WSStreaming |      False
16051 |     4 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     9.9178 |    9.4203 |     " H" | WSStreaming |      False
17412 |     4 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     9.9178 |   13.6559 |     " H" |     WSReady |      False
18949 |     4 | Classifier     |    True |   False |    False |   False |    False |    False |     9.9178 |   13.6559 |     " H" |     WSReady |      False
18950 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     9.9178 |   13.6559 |      "i" |     WSReady |       True
18976 |     0 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     9.9178 |   13.6559 |      "i" | WSStreaming |      False
19177 |     0 | ProcessingLayer |   False |   False |     True |   False |    False |    False |     2.7815 |   13.6559 |      "i" | WSStreaming |      False
20000 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |    False |     2.7815 |   13.6559 |      "i" |     WSReady |      False
20538 |     0 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     2.7815 |    2.8229 |      "i" |     WSReady |      False
20539 |     1 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     2.7815 |    2.8229 |      "i" |     WSReady |       True
20565 |     1 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     2.7815 |    2.8229 |      "i" | WSStreaming |      False
20766 |     1 | ProcessingLayer |   False |   False |     True |   False |    False |    False |     3.2277 |    2.8229 |      "i" | WSStreaming |      False
22127 |     1 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     3.2277 |    4.6433 |      "i" |     WSReady |      False
22128 |     2 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     3.2277 |    4.6433 |      "i" |     WSReady |       True
22154 |     2 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     3.2277 |    4.6433 |      "i" | WSStreaming |      False
22355 |     2 | ProcessingLayer |   False |   False |     True |   False |    False |    False |     5.5058 |    4.6433 |      "i" | WSStreaming |      False
23716 |     2 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     5.5058 |    6.1808 |      "i" |     WSReady |      False
23717 |     3 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     5.5058 |    6.1808 |      "i" |     WSReady |       True
23743 |     3 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     5.5058 |    6.1808 |      "i" | WSStreaming |      False
23944 |     3 | ProcessingLayer |   False |   False |     True |   False |    False |    False |     7.5363 |    6.1808 |      "i" | WSStreaming |      False
25305 |     3 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     7.5363 |    9.7465 |      "i" |     WSReady |      False
25306 |     4 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     7.5363 |    9.7465 |      "i" |     WSReady |       True
25332 |     4 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     7.5363 |    9.7465 |      "i" | WSStreaming |      False
25533 |     4 | ProcessingLayer |   False |   False |     True |   False |    False |    False |    10.3030 |    9.7465 |      "i" | WSStreaming |      False
26894 |     4 | ProcessingLayer |   False |   False |    False |    True |    False |    False |    10.3030 |   12.8257 |      "i" |     WSReady |      False
28431 |     4 | Classifier     |    True |   False |    False |   False |    False |    False |    10.3030 |   12.8257 |      "i" |     WSReady |      False
28432 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |     True |    10.3030 |   12.8257 |      "b" |     WSReady |       True
28458 |     0 | ProcessingLayer |   False |    True |    False |   False |     True |    False |    10.3030 |   12.8257 |      "b" | WSStreaming |      False
28660 |     0 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     2.8134 |   12.8257 |      "b" | WSStreaming |      False
30000 |     0 | ProcessingLayer |   False |   False |    False |   False |    False |    False |     2.8134 |   12.8257 |      "b" |     WSReady |      False
30021 |     0 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     2.8134 |    2.9711 |      "b" |     WSReady |      False
30022 |     1 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     2.8134 |    2.9711 |      "b" |     WSReady |       True
30048 |     1 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     2.8134 |    2.9711 |      "b" | WSStreaming |      False
30250 |     1 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     3.4127 |    2.9711 |      "b" | WSStreaming |      False
31611 |     1 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     3.4127 |    4.1652 |      "b" |     WSReady |      False
31612 |     2 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     3.4127 |    4.1652 |      "b" |     WSReady |       True
31638 |     2 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     3.4127 |    4.1652 |      "b" | WSStreaming |      False
31840 |     2 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     4.9870 |    4.1652 |      "b" | WSStreaming |      False
33201 |     2 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     4.9870 |    6.6421 |      "b" |     WSReady |      False
33202 |     3 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     4.9870 |    6.6421 |      "b" |     WSReady |       True
33228 |     3 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     4.9870 |    6.6421 |      "b" | WSStreaming |      False
33430 |     3 | ProcessingLayer |   False |   False |     True |   False |     True |    False |     8.1826 |    6.6421 |      "b" | WSStreaming |      False
34791 |     3 | ProcessingLayer |   False |   False |    False |    True |    False |    False |     8.1826 |    9.7846 |      "b" |     WSReady |      False
34792 |     4 | ProcessingLayer |   False |   False |    False |   False |    False |     True |     8.1826 |    9.7846 |      "b" |     WSReady |       True
34818 |     4 | ProcessingLayer |   False |    True |    False |   False |     True |    False |     8.1826 |    9.7846 |      "b" | WSStreaming |      False
35020 |     4 | ProcessingLayer |   False |   False |     True |   False |     True |    False |    10.1876 |    9.7846 |      "b" | WSStreaming |      False
36381 |     4 | ProcessingLayer |   False |   False |    False |    True |    False |    False |    10.1876 |   11.5531 |      "b" |     WSReady |      False
```
