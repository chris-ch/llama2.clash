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

Cycle | Layer | Stage              | Tok Rdy | QKVDone  | AttnDone  | FFNDone  | WgtValid | LayerChg | norm(attn) | norm(out) |    Tok    |   SsyState  | ddrWValid | ddrWReady | ddrBValid
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
    0 |     0 | Stage1_ProjectQKV  |   False |    False |     True |    False |    False |    False |        0.0 |        0.0 | "\n<s>\n" |     WSReady |     False |     False |     False
    1 |     0 | Stage1_ProjectQKV  |   False |    False |     True |    False |    False |    False |        0.0 |        0.0 | "\n<s>\n" |     WSReady |     False |     False |     False
   27 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |        0.0 |        0.0 | "\n<s>\n" | WSStreaming |     False |     False |     False
  227 |     0 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |   3.038473 |        0.0 | "\n<s>\n" | WSStreaming |     False |     False |     False
 1589 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |   3.038473 |   8.235682 | "\n<s>\n" |     WSReady |     False |     False |     False
 1590 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |   3.038473 |   8.235682 | "\n<s>\n" |     WSReady |     False |     False |     False
 1616 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |   3.038473 |   8.235682 | "\n<s>\n" | WSStreaming |     False |     False |     False
 1816 |     1 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |   9.176485 |   8.235682 | "\n<s>\n" | WSStreaming |     False |     False |     False
 3178 |     1 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |   9.176485 |  10.382461 | "\n<s>\n" |     WSReady |     False |     False |     False
 3179 |     2 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |   9.176485 |  10.382461 | "\n<s>\n" |     WSReady |     False |     False |     False
 3205 |     2 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |   9.176485 |  10.382461 | "\n<s>\n" | WSStreaming |     False |     False |     False
 3405 |     2 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  11.258031 |  10.382461 | "\n<s>\n" | WSStreaming |     False |     False |     False
 4767 |     2 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  11.258031 |  12.252162 | "\n<s>\n" |     WSReady |     False |     False |     False
 4768 |     3 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  11.258031 |  12.252162 | "\n<s>\n" |     WSReady |     False |     False |     False
 4794 |     3 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  11.258031 |  12.252162 | "\n<s>\n" | WSStreaming |     False |     False |     False
 4994 |     3 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  14.736917 |  12.252162 | "\n<s>\n" | WSStreaming |     False |     False |     False
 6356 |     3 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  14.736917 |  16.039158 | "\n<s>\n" |     WSReady |     False |     False |     False
 6357 |     4 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  14.736917 |  16.039158 | "\n<s>\n" |     WSReady |     False |     False |     False
 6383 |     4 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  14.736917 |  16.039158 | "\n<s>\n" | WSStreaming |     False |     False |     False
 6583 |     4 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  16.443026 |  16.039158 | "\n<s>\n" | WSStreaming |     False |     False |     False
 7945 |     4 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  16.443026 |  18.754534 | "\n<s>\n" |     WSReady |     False |     False |     False
 9482 |     4 | Stage5_Classifier  |    True |    False |    False |    False |    False |    False |  16.443026 |  18.754534 | "\n<s>\n" |     WSReady |     False |     False |     False
 9483 |     0 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  16.443026 |  18.754534 |     " H" |     WSReady |     False |     False |     False
 9509 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  16.443026 |  18.754534 |     " H" | WSStreaming |     False |     False |     False
 9710 |     0 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |  2.7742534 |  18.754534 |     " H" | WSStreaming |     False |     False |     False
10000 |     0 | Stage4_FeedForward |   False |    False |    False |    False |    False |    False |  2.7742534 |  16.481407 |     " H" | WSStreaming |     False |     False |     False
11072 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  2.7742534 |  3.8518157 |     " H" |     WSReady |     False |     False |     False
11073 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  2.7742534 |  3.8518157 |     " H" |     WSReady |     False |     False |     False
11099 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  2.7742534 |  3.8518157 |     " H" | WSStreaming |     False |     False |     False
11300 |     1 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |  4.5273285 |  3.8518157 |     " H" | WSStreaming |     False |     False |     False
12662 |     1 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  4.5273285 |  5.3426757 |     " H" |     WSReady |     False |     False |     False
12663 |     2 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  4.5273285 |  5.3426757 |     " H" |     WSReady |     False |     False |     False
12689 |     2 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  4.5273285 |  5.3426757 |     " H" | WSStreaming |     False |     False |     False
12890 |     2 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |  6.4986663 |  5.3426757 |     " H" | WSStreaming |     False |     False |     False
14252 |     2 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  6.4986663 |  6.8089294 |     " H" |     WSReady |     False |     False |     False
14253 |     3 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  6.4986663 |  6.8089294 |     " H" |     WSReady |     False |     False |     False
14279 |     3 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  6.4986663 |  6.8089294 |     " H" | WSStreaming |     False |     False |     False
14480 |     3 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |   8.510305 |  6.8089294 |     " H" | WSStreaming |     False |     False |     False
15842 |     3 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |   8.510305 |    9.42028 |     " H" |     WSReady |     False |     False |     False
15843 |     4 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |   8.510305 |    9.42028 |     " H" |     WSReady |     False |     False |     False
15869 |     4 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |   8.510305 |    9.42028 |     " H" | WSStreaming |     False |     False |     False
16070 |     4 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |   9.917849 |    9.42028 |     " H" | WSStreaming |     False |     False |     False
17432 |     4 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |   9.917849 |  13.655915 |     " H" |     WSReady |     False |     False |     False
18969 |     4 | Stage5_Classifier  |    True |    False |    False |    False |    False |    False |   9.917849 |  13.655915 |     " H" |     WSReady |     False |     False |     False
18970 |     0 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |   9.917849 |  13.655915 |      "i" |     WSReady |     False |     False |     False
18996 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |   9.917849 |  13.655915 |      "i" | WSStreaming |     False |     False |     False
19198 |     0 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  2.7814713 |  13.655915 |      "i" | WSStreaming |     False |     False |     False
20000 |     0 | Stage4_FeedForward |   False |    False |    False |    False |    False |    False |  2.7814713 |   9.014496 |      "i" |     WSReady |     False |     False |     False
20560 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  2.7814713 |   2.822887 |      "i" |     WSReady |     False |     False |     False
20561 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  2.7814713 |   2.822887 |      "i" |     WSReady |     False |     False |     False
20587 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  2.7814713 |   2.822887 |      "i" | WSStreaming |     False |     False |     False
20789 |     1 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  3.2277346 |   2.822887 |      "i" | WSStreaming |     False |     False |     False
22151 |     1 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  3.2277346 |  4.6433043 |      "i" |     WSReady |     False |     False |     False
22152 |     2 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  3.2277346 |  4.6433043 |      "i" |     WSReady |     False |     False |     False
22178 |     2 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  3.2277346 |  4.6433043 |      "i" | WSStreaming |     False |     False |     False
22380 |     2 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  5.5058446 |  4.6433043 |      "i" | WSStreaming |     False |     False |     False
23742 |     2 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  5.5058446 |   6.180796 |      "i" |     WSReady |     False |     False |     False
23743 |     3 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  5.5058446 |   6.180796 |      "i" |     WSReady |     False |     False |     False
23769 |     3 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  5.5058446 |   6.180796 |      "i" | WSStreaming |     False |     False |     False
23971 |     3 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  7.5363054 |   6.180796 |      "i" | WSStreaming |     False |     False |     False
25333 |     3 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  7.5363054 |   9.746549 |      "i" |     WSReady |     False |     False |     False
25334 |     4 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  7.5363054 |   9.746549 |      "i" |     WSReady |     False |     False |     False
25360 |     4 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  7.5363054 |   9.746549 |      "i" | WSStreaming |     False |     False |     False
25562 |     4 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  10.303003 |   9.746549 |      "i" | WSStreaming |     False |     False |     False
26924 |     4 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  10.303003 | 12.8256645 |      "i" |     WSReady |     False |     False |     False
28461 |     4 | Stage5_Classifier  |    True |    False |    False |    False |    False |    False |  10.303003 | 12.8256645 |      "i" |     WSReady |     False |     False |     False
28462 |     0 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  10.303003 | 12.8256645 |      "b" |     WSReady |     False |     False |     False
28488 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  10.303003 | 12.8256645 |      "b" | WSStreaming |     False |     False |     False
28691 |     0 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |   2.813378 | 12.8256645 |      "b" | WSStreaming |     False |     False |     False
30000 |     0 | Stage4_FeedForward |   False |    False |    False |    False |    False |    False |   2.813378 |   9.883355 |      "b" |     WSReady |     False |     False |     False
30053 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |   2.813378 |  2.9710772 |      "b" |     WSReady |     False |     False |     False
30054 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |   2.813378 |  2.9710772 |      "b" |     WSReady |     False |     False |     False
30080 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |   2.813378 |  2.9710772 |      "b" | WSStreaming |     False |     False |     False
30283 |     1 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |   3.412678 |  2.9710772 |      "b" | WSStreaming |     False |     False |     False
31645 |     1 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |   3.412678 |  4.1651764 |      "b" |     WSReady |     False |     False |     False
31646 |     2 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |   3.412678 |  4.1651764 |      "b" |     WSReady |     False |     False |     False
31672 |     2 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |   3.412678 |  4.1651764 |      "b" | WSStreaming |     False |     False |     False
31875 |     2 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |  4.9870014 |  4.1651764 |      "b" | WSStreaming |     False |     False |     False
33237 |     2 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  4.9870014 |   6.642122 |      "b" |     WSReady |     False |     False |     False
33238 |     3 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  4.9870014 |   6.642122 |      "b" |     WSReady |     False |     False |     False
33264 |     3 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  4.9870014 |   6.642122 |      "b" | WSStreaming |     False |     False |     False
33467 |     3 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |   8.182645 |   6.642122 |      "b" | WSStreaming |     False |     False |     False
34829 |     3 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |   8.182645 |   9.784572 |      "b" |     WSReady |     False |     False |     False
34830 |     4 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |   8.182645 |   9.784572 |      "b" |     WSReady |     False |     False |     False
34856 |     4 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |   8.182645 |   9.784572 |      "b" | WSStreaming |     False |     False |     False
35059 |     4 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |  10.187632 |   9.784572 |      "b" | WSStreaming |     False |     False |     False
36421 |     4 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  10.187632 |  11.553102 |      "b" |     WSReady |     False |     False |     False
37958 |     4 | Stage5_Classifier  |    True |    False |    False |    False |    False |    False |  10.187632 |  11.553102 |      "b" |     WSReady |     False |     False |     False
37959 |     0 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  10.187632 |  11.553102 |      "o" |     WSReady |     False |     False |     False
37985 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  10.187632 |  11.553102 |      "o" | WSStreaming |     False |     False |     False
38189 |     0 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  2.9592562 |  11.553102 |      "o" | WSStreaming |     False |     False |     False
39551 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  2.9592562 |  2.8445544 |      "o" |     WSReady |     False |     False |     False
39552 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  2.9592562 |  2.8445544 |      "o" |     WSReady |     False |     False |     False
39578 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  2.9592562 |  2.8445544 |      "o" | WSStreaming |     False |     False |     False
39782 |     1 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |  3.1207356 |  2.8445544 |      "o" | WSStreaming |     False |     False |     False
40000 |     1 | Stage4_FeedForward |   False |    False |    False |    False |     True |    False |  3.1207356 |  3.0059326 |      "o" | WSStreaming |     False |     False |     False
41144 |     1 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |  3.1207356 |  3.5186217 |      "o" |     WSReady |     False |     False |     False
41145 |     2 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |  3.1207356 |  3.5186217 |      "o" |     WSReady |     False |     False |     False
41171 |     2 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |  3.1207356 |  3.5186217 |      "o" | WSStreaming |     False |     False |     False
41375 |     2 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |   4.546975 |  3.5186217 |      "o" | WSStreaming |     False |     False |     False
```
