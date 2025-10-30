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

Cycle | Layer | Stage              | Tok Rdy | QKVDone  | AttnDone  | FFNDone  | WgtValid | LayerChg | WgtSmpl | norm(out) |    Tok    |   SsyState  | ddrWValid | ddrWReady | ddrBValid
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Loading: ./data/stories260K.bin
✅ Simulation Model (FPGA wired) Loaded
    0 |     0 | Stage1_ProjectQKV  |   False |    False |     True |    False |    False |    False |       0 |       0.0 | "\n<s>\n" |     WSReady |     False |     False |     False
    1 |     0 | Stage1_ProjectQKV  |   False |    False |     True |    False |    False |    False |       0 |       0.0 | "\n<s>\n" |     WSReady |     False |     False |     False
   27 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 |       0.0 | "\n<s>\n" | WSStreaming |     False |     False |     False
   28 |     0 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 |       0.0 | "\n<s>\n" | WSStreaming |     False |     False |     False
  227 |     0 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |       0 |       0.0 | "\n<s>\n" | WSStreaming |     False |     False |     False
  423 |     0 | Stage4_FeedForward |   False |    False |     True |    False |     True |    False |       0 |       0.0 | "\n<s>\n" | WSStreaming |     False |     False |     False
 1588 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       0 |  8.235682 | "\n<s>\n" |     WSReady |     False |     False |     False
 1589 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       0 |  8.235682 | "\n<s>\n" |     WSReady |     False |     False |     False
 1615 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 |  8.235682 | "\n<s>\n" | WSStreaming |     False |     False |     False
 1616 |     1 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 |  8.235682 | "\n<s>\n" | WSStreaming |     False |     False |     False
 1815 |     1 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |       0 |  8.235682 | "\n<s>\n" | WSStreaming |     False |     False |     False
 2011 |     1 | Stage4_FeedForward |   False |    False |     True |    False |     True |    False |       0 |  8.235682 | "\n<s>\n" | WSStreaming |     False |     False |     False
 3176 |     1 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       0 | 10.382461 | "\n<s>\n" |     WSReady |     False |     False |     False
 3177 |     2 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       0 | 10.382461 | "\n<s>\n" |     WSReady |     False |     False |     False
 3203 |     2 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 | 10.382461 | "\n<s>\n" | WSStreaming |     False |     False |     False
 3204 |     2 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 10.382461 | "\n<s>\n" | WSStreaming |     False |     False |     False
 3403 |     2 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |       0 | 10.382461 | "\n<s>\n" | WSStreaming |     False |     False |     False
 3599 |     2 | Stage4_FeedForward |   False |    False |     True |    False |     True |    False |       0 | 10.382461 | "\n<s>\n" | WSStreaming |     False |     False |     False
 4764 |     2 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       0 | 12.252162 | "\n<s>\n" |     WSReady |     False |     False |     False
 4765 |     3 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       0 | 12.252162 | "\n<s>\n" |     WSReady |     False |     False |     False
 4791 |     3 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 | 12.252162 | "\n<s>\n" | WSStreaming |     False |     False |     False
 4792 |     3 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 12.252162 | "\n<s>\n" | WSStreaming |     False |     False |     False
 4991 |     3 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |       0 | 12.252162 | "\n<s>\n" | WSStreaming |     False |     False |     False
 5187 |     3 | Stage4_FeedForward |   False |    False |     True |    False |     True |    False |       0 | 12.252162 | "\n<s>\n" | WSStreaming |     False |     False |     False
 6352 |     3 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |     -42 | 16.039158 | "\n<s>\n" |     WSReady |     False |     False |     False
 6353 |     4 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |     -42 | 16.039158 | "\n<s>\n" |     WSReady |     False |     False |     False
 6379 |     4 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 | 16.039158 | "\n<s>\n" | WSStreaming |     False |     False |     False
 6380 |     4 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 16.039158 | "\n<s>\n" | WSStreaming |     False |     False |     False
 6579 |     4 | Stage3_Attend      |   False |    False |     True |    False |     True |    False |       0 | 16.039158 | "\n<s>\n" | WSStreaming |     False |     False |     False
 6775 |     4 | Stage4_FeedForward |   False |    False |     True |    False |     True |    False |       0 | 16.039158 | "\n<s>\n" | WSStreaming |     False |     False |     False
 7940 |     4 | Stage4_FeedForward |    True |    False |    False |     True |    False |    False |       0 | 18.754534 | "\n<s>\n" |     WSReady |     False |     False |     False
 7941 |     0 | Stage5_Classifier  |   False |    False |    False |    False |    False |     True |       0 | 18.754534 |     " H" |     WSReady |     False |     False |     False
 9505 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |    False |    False |       0 | 18.754534 |     " H" |     WSReady |     False |     False |     False
 9506 |     0 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 18.754534 |     " H" |     WSReady |     False |     False |     False
 9706 |     0 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |       0 | 18.754534 |     " H" |     WSReady |     False |     False |     False
 9902 |     0 | Stage4_FeedForward |   False |    False |     True |    False |    False |    False |       0 | 18.754534 |     " H" |     WSReady |     False |     False |     False
10000 |     0 | Stage4_FeedForward |   False |    False |    False |    False |    False |    False |       0 | 18.754534 |     " H" |     WSReady |     False |     False |     False
11067 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       0 | 3.8518157 |     " H" |     WSReady |     False |     False |     False
11068 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       0 | 3.8518157 |     " H" |     WSReady |     False |     False |     False
11094 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 | 3.8518157 |     " H" | WSStreaming |     False |     False |     False
11095 |     1 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 3.8518157 |     " H" | WSStreaming |     False |     False |     False
11295 |     1 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |       0 | 3.8518157 |     " H" | WSStreaming |     False |     False |     False
11491 |     1 | Stage4_FeedForward |   False |    False |     True |    False |    False |    False |       0 | 3.8518157 |     " H" | WSStreaming |     False |     False |     False
12656 |     1 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       0 | 5.3426757 |     " H" |     WSReady |     False |     False |     False
12657 |     2 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       0 | 5.3426757 |     " H" |     WSReady |     False |     False |     False
12683 |     2 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 | 5.3426757 |     " H" | WSStreaming |     False |     False |     False
12684 |     2 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 5.3426757 |     " H" | WSStreaming |     False |     False |     False
12884 |     2 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |       0 | 5.3426757 |     " H" | WSStreaming |     False |     False |     False
13080 |     2 | Stage4_FeedForward |   False |    False |     True |    False |    False |    False |       0 | 5.3426757 |     " H" | WSStreaming |     False |     False |     False
14245 |     2 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       0 | 6.8089294 |     " H" |     WSReady |     False |     False |     False
14246 |     3 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       0 | 6.8089294 |     " H" |     WSReady |     False |     False |     False
14272 |     3 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 | 6.8089294 |     " H" | WSStreaming |     False |     False |     False
14273 |     3 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 6.8089294 |     " H" | WSStreaming |     False |     False |     False
14473 |     3 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |       0 | 6.8089294 |     " H" | WSStreaming |     False |     False |     False
14669 |     3 | Stage4_FeedForward |   False |    False |     True |    False |    False |    False |       0 | 6.8089294 |     " H" | WSStreaming |     False |     False |     False
15834 |     3 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       8 |   9.42028 |     " H" |     WSReady |     False |     False |     False
15835 |     4 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       8 |   9.42028 |     " H" |     WSReady |     False |     False |     False
15861 |     4 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 |   9.42028 |     " H" | WSStreaming |     False |     False |     False
15862 |     4 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 |   9.42028 |     " H" | WSStreaming |     False |     False |     False
16062 |     4 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |       0 |   9.42028 |     " H" | WSStreaming |     False |     False |     False
16258 |     4 | Stage4_FeedForward |   False |    False |     True |    False |    False |    False |       0 |   9.42028 |     " H" | WSStreaming |     False |     False |     False
17423 |     4 | Stage4_FeedForward |    True |    False |    False |     True |    False |    False |       0 | 13.655915 |     " H" |     WSReady |     False |     False |     False
17424 |     0 | Stage5_Classifier  |   False |    False |    False |    False |    False |     True |       0 | 13.655915 |      "i" |     WSReady |     False |     False |     False
18988 |     0 | Stage1_ProjectQKV  |   False |     True |    False |    False |    False |    False |       0 | 13.655915 |      "i" |     WSReady |     False |     False |     False
18989 |     0 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 | 13.655915 |      "i" |     WSReady |     False |     False |     False
19190 |     0 | Stage3_Attend      |   False |    False |     True |    False |    False |    False |       0 | 13.655915 |      "i" |     WSReady |     False |     False |     False
19386 |     0 | Stage4_FeedForward |   False |    False |     True |    False |    False |    False |       0 | 13.655915 |      "i" |     WSReady |     False |     False |     False
20000 |     0 | Stage4_FeedForward |   False |    False |    False |    False |    False |    False |       0 | 13.655915 |      "i" |     WSReady |     False |     False |     False
20551 |     0 | Stage4_FeedForward |   False |    False |    False |     True |    False |    False |       0 |  2.822887 |      "i" |     WSReady |     False |     False |     False
20552 |     1 | Stage1_ProjectQKV  |   False |    False |    False |    False |    False |     True |       0 |  2.822887 |      "i" |     WSReady |     False |     False |     False
20578 |     1 | Stage1_ProjectQKV  |   False |     True |    False |    False |     True |    False |       0 |  2.822887 |      "i" | WSStreaming |     False |     False |     False
20579 |     1 | Stage2_WriteKV     |   False |     True |    False |    False |    False |    False |       0 |  2.822887 |      "i" | WSStreaming |     False |     False |     False
```
