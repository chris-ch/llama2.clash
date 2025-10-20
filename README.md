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

# Few first cycles (combinational multiplier)

```text

Prepending BOS to prompt tokens: [1,320,417]
✅ model loaded successfully
✅ Prompt: [1,320,417]

Cycle | Layer | Stage | Ready | AttnDone | FFNDone | Token
-----------------------------------------------------------
    0 |     0 | Stage1_ProjectQKV                | False |     True |    False |     1
   19 |     0 | Stage4_FeedForward               | False |    False |     True |     1
   38 |     1 | Stage4_FeedForward               | False |    False |     True |     1
   57 |     2 | Stage4_FeedForward               | False |    False |     True |     1
   76 |     3 | Stage4_FeedForward               | False |    False |     True |     1
   95 |     4 | Stage4_FeedForward               | False |    False |     True |     1
   96 |     4 | Stage5_Classifier                |  True |    False |     True |     1
  117 |     0 | Stage4_FeedForward               | False |    False |     True |   320
  137 |     1 | Stage4_FeedForward               | False |    False |     True |   320
  157 |     2 | Stage4_FeedForward               | False |    False |     True |   320
  177 |     3 | Stage4_FeedForward               | False |    False |     True |   320
  197 |     4 | Stage4_FeedForward               | False |    False |     True |   320
  198 |     4 | Stage5_Classifier                |  True |    False |     True |   320
  220 |     0 | Stage4_FeedForward               | False |    False |     True |   417
  241 |     1 | Stage4_FeedForward               | False |    False |     True |   417
  262 |     2 | Stage4_FeedForward               | False |    False |     True |   417
  283 |     3 | Stage4_FeedForward               | False |    False |     True |   417
  304 |     4 | Stage4_FeedForward               | False |    False |     True |   417
  305 |     4 | Stage5_Classifier                |  True |    False |     True |   417
  328 |     0 | Stage4_FeedForward               | False |    False |     True |   430
  350 |     1 | Stage4_FeedForward               | False |    False |     True |   430
  372 |     2 | Stage4_FeedForward               | False |    False |     True |   430
  394 |     3 | Stage4_FeedForward               | False |    False |     True |   430
  416 |     4 | Stage4_FeedForward               | False |    False |     True |   430
  417 |     4 | Stage5_Classifier                |  True |    False |     True |   430
  441 |     0 | Stage4_FeedForward               | False |    False |     True |   414
  464 |     1 | Stage4_FeedForward               | False |    False |     True |   414
  487 |     2 | Stage4_FeedForward               | False |    False |     True |   414
  510 |     3 | Stage4_FeedForward               | False |    False |     True |   414
  533 |     4 | Stage4_FeedForward               | False |    False |     True |   414
  534 |     4 | Stage5_Classifier                |  True |    False |     True |   414
  559 |     0 | Stage4_FeedForward               | False |    False |     True |   269
  583 |     1 | Stage4_FeedForward               | False |    False |     True |   269
  607 |     2 | Stage4_FeedForward               | False |    False |     True |   269
  631 |     3 | Stage4_FeedForward               | False |    False |     True |   269
  655 |     4 | Stage4_FeedForward               | False |    False |     True |   269
  656 |     4 | Stage5_Classifier                |  True |    False |     True |   269
  682 |     0 | Stage4_FeedForward               | False |    False |     True |   410
  707 |     1 | Stage4_FeedForward               | False |    False |     True |   410
  732 |     2 | Stage4_FeedForward               | False |    False |     True |   410
  757 |     3 | Stage4_FeedForward               | False |    False |     True |   410
  782 |     4 | Stage4_FeedForward               | False |    False |     True |   410
  783 |     4 | Stage5_Classifier                |  True |    False |     True |   410
  810 |     0 | Stage4_FeedForward               | False |    False |     True |   447
  836 |     1 | Stage4_FeedForward               | False |    False |     True |   447
  862 |     2 | Stage4_FeedForward               | False |    False |     True |   447
  888 |     3 | Stage4_FeedForward               | False |    False |     True |   447
  914 |     4 | Stage4_FeedForward               | False |    False |     True |   447
  915 |     4 | Stage5_Classifier                |  True |    False |     True |   447
  943 |     0 | Stage4_FeedForward               | False |    False |     True |   416
  970 |     1 | Stage4_FeedForward               | False |    False |     True |   416
  997 |     2 | Stage4_FeedForward               | False |    False |     True |   416
 1000 |     3 | Stage1_ProjectQKV                | False |    False |    False |   416
 1024 |     3 | Stage4_FeedForward               | False |    False |     True |   416
 1051 |     4 | Stage4_FeedForward               | False |    False |     True |   416
 1052 |     4 | Stage5_Classifier                |  True |    False |     True |   416
 1081 |     0 | Stage4_FeedForward               | False |    False |     True |   416
 1109 |     1 | Stage4_FeedForward               | False |    False |     True |   416
 1137 |     2 | Stage4_FeedForward               | False |    False |     True |   416
 1165 |     3 | Stage4_FeedForward               | False |    False |     True |   416
 1193 |     4 | Stage4_FeedForward               | False |    False |     True |   416
 1194 |     4 | Stage5_Classifier                |  True |    False |     True |   416
 1224 |     0 | Stage4_FeedForward               | False |    False |     True |   412
 1253 |     1 | Stage4_FeedForward               | False |    False |     True |   412
 1282 |     2 | Stage4_FeedForward               | False |    False |     True |   412
 1311 |     3 | Stage4_FeedForward               | False |    False |     True |   412
 1340 |     4 | Stage4_FeedForward               | False |    False |     True |   412
 1341 |     4 | Stage5_Classifier                |  True |    False |     True |   412
 1372 |     0 | Stage4_FeedForward               | False |    False |     True |   432
 1402 |     1 | Stage4_FeedForward               | False |    False |     True |   432
 1432 |     2 | Stage4_FeedForward               | False |    False |     True |   432
 1462 |     3 | Stage4_FeedForward               | False |    False |     True |   432
 1492 |     4 | Stage4_FeedForward               | False |    False |     True |   432
 1493 |     4 | Stage5_Classifier                |  True |    False |     True |   432
 1525 |     0 | Stage4_FeedForward               | False |    False |     True |   410
 1556 |     1 | Stage4_FeedForward               | False |    False |     True |   410
 1587 |     2 | Stage4_FeedForward               | False |    False |     True |   410
 1618 |     3 | Stage4_FeedForward               | False |    False |     True |   410
 1649 |     4 | Stage4_FeedForward               | False |    False |     True |   410
 1650 |     4 | Stage5_Classifier                |  True |    False |     True |   410
 1683 |     0 | Stage4_FeedForward               | False |    False |     True |   447
 1715 |     1 | Stage4_FeedForward               | False |    False |     True |   447
 1747 |     2 | Stage4_FeedForward               | False |    False |     True |   447
 1779 |     3 | Stage4_FeedForward               | False |    False |     True |   447
 1811 |     4 | Stage4_FeedForward               | False |    False |     True |   447
 1812 |     4 | Stage5_Classifier                |  True |    False |     True |   447
 1846 |     0 | Stage4_FeedForward               | False |    False |     True |   416
 1879 |     1 | Stage4_FeedForward               | False |    False |     True |   416
 1912 |     2 | Stage4_FeedForward               | False |    False |     True |   416
 1945 |     3 | Stage4_FeedForward               | False |    False |     True |   416
 1978 |     4 | Stage4_FeedForward               | False |    False |     True |   416
 1979 |     4 | Stage5_Classifier                |  True |    False |     True |   416
 2000 |     0 | Stage3_Attend                    | False |    False |    False |   416
 2014 |     0 | Stage4_FeedForward               | False |    False |     True |   416
 2048 |     1 | Stage4_FeedForward               | False |    False |     True |   416
 2082 |     2 | Stage4_FeedForward               | False |    False |     True |   416
 2116 |     3 | Stage4_FeedForward               | False |    False |     True |   416
 2150 |     4 | Stage4_FeedForward               | False |    False |     True |   416
 2151 |     4 | Stage5_Classifier                |  True |    False |     True |   416
 2187 |     0 | Stage4_FeedForward               | False |    False |     True |   412
 2222 |     1 | Stage4_FeedForward               | False |    False |     True |   412
 2257 |     2 | Stage4_FeedForward               | False |    False |     True |   412
 2292 |     3 | Stage4_FeedForward               | False |    False |     True |   412
 2327 |     4 | Stage4_FeedForward               | False |    False |     True |   412
 2328 |     4 | Stage5_Classifier                |  True |    False |     True |   412
 2365 |     0 | Stage4_FeedForward               | False |    False |     True |   432
 2401 |     1 | Stage4_FeedForward               | False |    False |     True |   432
 2437 |     2 | Stage4_FeedForward               | False |    False |     True |   432
 2473 |     3 | Stage4_FeedForward               | False |    False |     True |   432
 2509 |     4 | Stage4_FeedForward               | False |    False |     True |   432
 2510 |     4 | Stage5_Classifier                |  True |    False |     True |   432
 2548 |     0 | Stage4_FeedForward               | False |    False |     True |   410
 2585 |     1 | Stage4_FeedForward               | False |    False |     True |   410
 2622 |     2 | Stage4_FeedForward               | False |    False |     True |   410
 2659 |     3 | Stage4_FeedForward               | False |    False |     True |   410
 2696 |     4 | Stage4_FeedForward               | False |    False |     True |   410
 2697 |     4 | Stage5_Classifier                |  True |    False |     True |   410
 2736 |     0 | Stage4_FeedForward               | False |    False |     True |   447
 2774 |     1 | Stage4_FeedForward               | False |    False |     True |   447
 2812 |     2 | Stage4_FeedForward               | False |    False |     True |   447
 2850 |     3 | Stage4_FeedForward               | False |    False |     True |   447
 2888 |     4 | Stage4_FeedForward               | False |    False |     True |   447
 2889 |     4 | Stage5_Classifier                |  True |    False |     True |   447
 2929 |     0 | Stage4_FeedForward               | False |    False |     True |   416
 2968 |     1 | Stage4_FeedForward               | False |    False |     True |   416
 3000 |     2 | Stage4_FeedForward               | False |    False |    False |   416
 3007 |     2 | Stage4_FeedForward               | False |    False |     True |   416
 3046 |     3 | Stage4_FeedForward               | False |    False |     True |   416
 3085 |     4 | Stage4_FeedForward               | False |    False |     True |   416
 3086 |     4 | Stage5_Classifier                |  True |    False |     True |   416
 3127 |     0 | Stage4_FeedForward               | False |    False |     True |   416
 3167 |     1 | Stage4_FeedForward               | False |    False |     True |   416
 3207 |     2 | Stage4_FeedForward               | False |    False |     True |   416
 3247 |     3 | Stage4_FeedForward               | False |    False |     True |   416
 3287 |     4 | Stage4_FeedForward               | False |    False |     True |   416
 3288 |     4 | Stage5_Classifier                |  True |    False |     True |   416
 3330 |     0 | Stage4_FeedForward               | False |    False |     True |   412
 3371 |     1 | Stage4_FeedForward               | False |    False |     True |   412
 3412 |     2 | Stage4_FeedForward               | False |    False |     True |   412
 3453 |     3 | Stage4_FeedForward               | False |    False |     True |   412
 3494 |     4 | Stage4_FeedForward               | False |    False |     True |   412
 3495 |     4 | Stage5_Classifier                |  True |    False |     True |   412
 3538 |     0 | Stage4_FeedForward               | False |    False |     True |   432
 3580 |     1 | Stage4_FeedForward               | False |    False |     True |   432
 3622 |     2 | Stage4_FeedForward               | False |    False |     True |   432
 3664 |     3 | Stage4_FeedForward               | False |    False |     True |   432
 3706 |     4 | Stage4_FeedForward               | False |    False |     True |   432
 3707 |     4 | Stage5_Classifier                |  True |    False |     True |   432
 3751 |     0 | Stage4_FeedForward               | False |    False |     True |   280
 3794 |     1 | Stage4_FeedForward               | False |    False |     True |   280
 3837 |     2 | Stage4_FeedForward               | False |    False |     True |   280
 3880 |     3 | Stage4_FeedForward               | False |    False |     True |   280
 3923 |     4 | Stage4_FeedForward               | False |    False |     True |   280
 3924 |     4 | Stage5_Classifier                |  True |    False |     True |   280
 3969 |     0 | Stage4_FeedForward               | False |    False |     True |   314
 4000 |     1 | Stage3_Attend                    | False |    False |    False |   314
 4013 |     1 | Stage4_FeedForward               | False |    False |     True |   314
 4057 |     2 | Stage4_FeedForward               | False |    False |     True |   314
 4101 |     3 | Stage4_FeedForward               | False |    False |     True |   314
 4145 |     4 | Stage4_FeedForward               | False |    False |     True |   314
 4146 |     4 | Stage5_Classifier                |  True |    False |     True |   314
 4192 |     0 | Stage4_FeedForward               | False |    False |     True |   411
 4237 |     1 | Stage4_FeedForward               | False |    False |     True |   411
 4282 |     2 | Stage4_FeedForward               | False |    False |     True |   411
 4327 |     3 | Stage4_FeedForward               | False |    False |     True |   411
 4372 |     4 | Stage4_FeedForward               | False |    False |     True |   411
 4373 |     4 | Stage5_Classifier                |  True |    False |     True |   411
 4420 |     0 | Stage4_FeedForward               | False |    False |     True |   267
 4466 |     1 | Stage4_FeedForward               | False |    False |     True |   267
 4512 |     2 | Stage4_FeedForward               | False |    False |     True |   267
 4558 |     3 | Stage4_FeedForward               | False |    False |     True |   267
 4604 |     4 | Stage4_FeedForward               | False |    False |     True |   267
 4605 |     4 | Stage5_Classifier                |  True |    False |     True |   267

```
