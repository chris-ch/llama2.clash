# LLaMA‑2 Decoder (Clash) — Plan and Tasks

Scope
- Target: synthesizable, timing‑clean hardware decoder for LLaMA‑2 style transformer in Clash.
- Goal: keep current API/structure where possible; replace impractical blocks with streaming/pipelined variants.

# Design choices and constraints

## Toolchain and scope
 - Clash 1.8.2, GHC 9.2.8. Synthesizable only; no floating point in the final design.
 - Target device flexible; prioritize simplicity and timing robustness over maximum clock.
 - Goal: LLaMA2/3-class decoder, context length > 16k.

# GPT-5 Notes

Reasoning, assumptions, and approach
- Goal: make the design realistically synthesizable on an FPGA in Clash 1.8.2 / GHC 9.6.7, fixed-point only on datapath. Your current code is functionally clean, but several blocks are still “algorithmic” (combinational Vec math over large dimensions) and/or memory-unrealistic (Vec ROMs feeding huge muxes).
- I audited every module you posted with these questions in mind:
  1) Will Clash infer BRAMs and small SRAMs instead of giant mux nets?
  2) Are long combinational paths broken up or time-multiplexed?
  3) Does every multi-cycle stage have a “done” handshake so the pipeline controller can actually wait?
  4) Are all registers initialized to defined values?
  5) Is the top entity elaboration-friendly (no higher-order args, deterministically monomorphic, annotated)?
- Summary: you are close on the sequential attention and KV-cache pieces, but Stage1 (QKV projection), vocab embedding and the FFN are still fully combinational. Also, the top-level needs a synthesizable wrapper, a Stage1 “done” handshake in the pipeline controller, and a few utilities need minor fixes for reset semantics and write-pulsing.

What remains to be done (blockers vs improvements)
Blockers (must-fix to build a real bitstream)
1) Top entity: remove higher‑order parameter at the hardware boundary and add a Synthesize annotation.
   - Today: topEntity :: TransformerDecoderComponent -> Signal … -> …
   - For synthesis, the decoder (weights, LUTs) must be a constant bound inside the top or provided via an external loader, not a port. Provide a new top that is fully applied and monomorphic.

2) Use true ROM/BRAM for the vocabulary table (and large parameter tables).
   - Model.Core.Embedding.embedder currently indexes a Vec with a runtime Int/Token. That becomes a giant mux and will not map to BRAM for large VocabularySize (e.g., 32k). Use rom/romPow2 (1‑cycle) or blockRam (1‑cycle) to force BRAM.

3) Stage1 (QKV projection) and FFN are purely combinational mat‑vec on full ModelDimension. That won’t close timing and creates huge multiplier farms.
   - Replace matrixVectorMult (combinational) with a sequential mat‑vec engine (you already have MatVecI8E_Seq.matVecRowSeq as a kernel) and add a Stage1 “done” handshake.
   - Do the same for FFN’s three mat‑vecs (W1, W3, W2). A single time‑multiplexed engine with a small controller can run all three passes.

4) Pipeline controller needs a Stage1 done input.
   - runPipelineController currently advances from Stage1 to Stage2 unconditionally (except for the very first token). You must hold at Stage1 until “projQKVDoneThisLayer == 1”.

5) KV write in Stage2 repeatedly writes the same row every cycle while Stage2 is active.
   - Gate the write with a 1‑cycle rising‑edge pulse so you write exactly once, then assert “writeDone” without wasting cycles or energy.

6) Register init Xs.
   - holdWhen uses deepErrorX for init. You want a defined reset value for synthesis.

Reasoning and approach
- Goal: add a synthesizable, fixed-point, single-engine sequential mat-vec for Q/K/V and FFN that reuses your existing 1-row kernel matVecRowSeq. We will:
  1) Build a tiny, deterministic row/column controller that asserts clear for 1 cycle, then drives en for exactly cols cycles with lastCol asserted on the final element. It also mirrors the kernel’s internal column counter so we can stream the input vector x by indexing with the same col index.
  2) Feed static per-row ROM data (RowI8E) into matVecRowSeq via a row selector that is latched at the row’s clear cycle, so the row is stable during accumulation.
  3) Accumulate the per-row results into a per-matrix work vector and, at pass boundaries, commit into the appropriate Q/K/V or FFN vectors.
  4) Time-multiplex one engine across passes: Q → K → V for Stage 1, and W1 → W3 → W2 for FFN.
  5) Emit a single s1Done pulse at the last Wv row, and a single ffnDone pulse at the last W2 row.
- Important dimensional note: for matrix-vector multiply, “columns” = ModelDimension (not HeadDimension). The bullet in your request that says “iterate columns for HeadDimension cycles” would be correct for Stage 3 attention (dot over HeadDimension), but for Q/K/V/FFN mat-vec the column count is the input dimension of each matrix (ModelDimension for Wq/Wk/Wv and W1/W3; HiddenDimension for W2).
- Interfaces: I keep interfaces as “start/done pulse + whole-vector outputs” so they can be wired into your pipeline easily. Stage-1 done replaces the stub in Transformer.multiCycleTransformer. If you want fully incremental writes, the commit points are called out.

New: generic row/column controller (1-row kernel driver)
File: project/llama2/Model/Helpers/Seq/RowColCtrl.hs
```haskell
{-# LANGUAGE ScopedTypeVariables #-}
module Model.Helpers.Seq.RowColCtrl
  ( RowColCtrlOut(..)
  , rowColCtrl
  ) where

import Clash.Prelude

-- Output bundle for one row drive and row scan
data RowColCtrlOut rows cols = RowColCtrlOut
  { clear    :: Signal System Bool              -- 1-cycle pulse at row start
  , en       :: Signal System Bool              -- advance one column when True
  , lastCol  :: Signal System Bool              -- asserted with en on final column
  , colIdx   :: Signal System (Index cols)      -- external copy of kernel's col
  , rowIdx   :: Signal System (Index rows)      -- current row (stable during row)
  , rowDone  :: Signal System Bool              -- 1-cycle pulse (next cycle after lastCol)
  , allDone  :: Signal System Bool              -- 1-cycle pulse at final row completion
  } deriving (Generic, NFDataX)

-- Drives one row at a time. Start is a 1-cycle pulse.
-- Protocol:
--  * clear high for 1 cycle at row start.
--  * en high for exactly 'cols' cycles (lastCol asserted on final en).
--  * rowDone pulses 1 cycle after lastCol; rowIdx is still the completed row on that cycle.
--  * When the final row completes, allDone pulses for 1 cycle and controller idles.
rowColCtrl
  :: forall rows cols
   . (KnownNat rows, KnownNat cols)
  => Signal System Bool               -- ^ start pulse
  -> RowColCtrlOut rows cols
rowColCtrl start = RowColCtrlOut{..}
 where
  -- FSM: Idle -> RowRun -> RowGap (commit) -> Idle (after last) or next RowRun
  data S = Idle | RowRun | RowGap deriving (Generic, NFDataX, Eq)

  -- Row index
  rowReg :: Signal System (Index rows)
  rowReg = mealy goRow 0 (bundle (state, rowDone))
   where
    goRow r (st, rd) =
      let r' = case st of
                 Idle   -> if unbundle start == (True, ) undefined then 0 else r
                 RowRun -> r
                 RowGap -> if rd then (if r == maxBound then r else succ r) else r
      in (r', r')
  rowIdx = rowReg

  -- State register
  state :: Signal System S
  state = mealy goS Idle (bundle (start, rowDone, rowIdx))
   where
    goS s (st, rd, r) =
      let s' = case s of
                 Idle   -> if st then RowRun else Idle
                 RowRun -> if rd then RowGap else RowRun
                 RowGap -> if r == maxBound then Idle else RowRun
      in (s', s')

  -- Column counter mirrors kernel
  colReg :: Signal System (Index cols)
  colReg = mealy goCol 0 (bundle (clear, en))
   where
    goCol c (cl, step) =
      let c0 = if cl then 0 else c
          cN = if step then (if c0 == maxBound then c0 else succ c0) else c0
      in (cN, c0)
  colIdx = colReg

  -- Control generation
  isRowRun = (== RowRun) <$> state
  en       = isRowRun
  clear    = regEn False isRowRun (pure True)  -- 1 cycle high when RowRun begins
  lastCol  = en .&&. ((== maxBound) <$> colIdx)
  rowDone  = regEn False en lastCol
  allDone  = rowDone .&&. ((== maxBound) <$> rowIdx)
```

Why it works
- The controller replicates the exact clear/en/lastCol contract expected by matVecRowSeq, ensuring the external colIdx stays aligned to the kernel’s internal counter. Row changes only occur in the RowGap state (the commit cycle), so at rowDone the rowIdx still points to the completed row.

QKV sequential engine over all heads with single kernel
File: project/llama2/Model/Layers/Attention/ProjectQKVSeq.hs
```haskell
{-# LANGUAGE ScopedTypeVariables #-}
module Model.Layers.Attention.ProjectQKVSeq
  ( projectQKVSeq
  , stage1ProjectQKVSeqLayer
  ) where

import Clash.Prelude
import Model.Config
  ( ModelDimension, HeadDimension
  , NumQueryHeads, NumKeyValueHeads
  , SequenceLength, NumLayers )
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D(..), RowI8E)
import Model.Helpers.MatVecI8E_Seq (matVecRowSeq)
import Model.Helpers.Seq.RowColCtrl (rowColCtrl, RowColCtrlOut(..))
import Model.Helpers.FixedPoint (rmsNormFwFix)
import Model.Layers.Components.Quantized (MultiHeadAttentionComponentQ(..), SingleHeadComponentQ(..))
import Model.Layers.Attention.MultiHeadAttention.Internal (applyRotation)
import Model.Core.Types (ProcessingState(..), CycleStage(..), LayerData(..))

-- Map each KV head to the query head it shares rotary/weight bank with
queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 kvIx =
  let nQ  = natToNum @NumQueryHeads
      nKV = natToNum @NumKeyValueHeads
      base = min (nQ - 1) (fromEnum kvIx * (nQ `div` nKV))
  in toEnum base

-- Select current row for the active pass/head/row
selectRowQ
  :: Vec NumQueryHeads (QArray2D HeadDimension ModelDimension)
  -> Index NumQueryHeads
  -> Index HeadDimension
  -> RowI8E ModelDimension
selectRowQ headsQ qIx rIx =
  let QArray2D m = (headsQ !! qIx)
  in m !! rIx

selectRowK
  :: Vec NumQueryHeads (QArray2D HeadDimension ModelDimension)
  -> Index NumKeyValueHeads
  -> Index HeadDimension
  -> RowI8E ModelDimension
selectRowK headsQ kvIx rIx =
  let qIx = queryHeadIndex0 kvIx
      QArray2D m = wkHeadQ (SingleHeadComponentQ (wqHeadQ (headsQ !! qIx)) (QArray2D (repeat (repeat (0,0)))) (QArray2D (repeat (repeat (0,0)))) (rotaryQ (headsQ !! qIx)))
  in m !! rIx

selectRowV
  :: Vec NumQueryHeads (QArray2D HeadDimension ModelDimension)
  -> Index NumKeyValueHeads
  -> Index HeadDimension
  -> RowI8E ModelDimension
selectRowV headsQ kvIx rIx =
  let qIx = queryHeadIndex0 kvIx
      QArray2D m = wvHeadQ (headsQ !! qIx)
  in m !! rIx

-- Engine state for time-multiplexed passes
data Pass = PassIdle | PassQ | PassK | PassV deriving (Generic, NFDataX, Eq, Show)

-- One shared sequential engine that computes:
--  - all NumQueryHeads query vectors (rotary applied)
--  - all NumKeyValueHeads key vectors (rotary applied)
--  - all NumKeyValueHeads value vectors (no rotary)
-- start is a 1-cycle pulse; done pulses after the last V row completes.
projectQKVSeq
  :: HiddenClockResetEnable System
  => MultiHeadAttentionComponentQ
  -> Signal System (Index SequenceLength)                    -- ^ seq pos for rotary
  -> Signal System (Vec ModelDimension FixedPoint)           -- ^ input x (already RMS-normalized preferred)
  -> Signal System Bool                                      -- ^ start pulse
  -> ( Signal System (Vec NumQueryHeads    (Vec HeadDimension FixedPoint))  -- Q
     , Signal System (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))  -- K
     , Signal System (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))  -- V
     , Signal System Bool )                                   -- done
projectQKVSeq mha seqPos xHat start = (qOut, kOut, vOut, allDonePass)
 where
  headsQv = headsQ mha

  -- Controller per row (rows=headDim, cols=modelDim)
  rc :: RowColCtrlOut HeadDimension ModelDimension
  rc = rowColCtrl start
  clearS  = clear rc
  enS     = en rc
  lastCol = lastCol rc
  colIdxS = colIdx rc
  rIdxS   = rowIdx rc
  rowDone = rowDone rc

  -- Pass/head iteration
  passReg :: Signal System Pass
  passReg = mealy goPass PassIdle (bundle (start, rowDone, rIdxS, qHeadReg, kvHeadReg))
   where
    goPass p (st, rd, r, qh, kvh) =
      let onRowLast = rd && r == maxBound
          p' = case p of
                 PassIdle -> if st then PassQ else PassIdle
                 PassQ    -> if onRowLast && qh == maxBound then PassK else PassQ
                 PassK    -> if onRowLast && kvh == maxBound then PassV else PassK
                 PassV    -> if onRowLast && kvh == maxBound then PassIdle else PassV
      in (p', p')

  -- head indices advance at head completion (last row done)
  qHeadReg :: Signal System (Index NumQueryHeads)
  qHeadReg = mealy goQ 0 (bundle (passReg, rowDone, rIdxS))
   where
    goQ qh (p, rd, r) =
      let qh' = case p of
                  PassQ | rd && r == maxBound -> if qh == maxBound then qh else succ qh
                  _                           -> qh
      in (qh', qh')

  kvHeadReg :: Signal System (Index NumKeyValueHeads)
  kvHeadReg = mealy goKV 0 (bundle (passReg, rowDone, rIdxS))
   where
    goKV kh (p, rd, r) =
      let kh' = case p of
                  PassK | rd && r == maxBound -> if kh == maxBound then kh else succ kh
                  PassV | rd && r == maxBound -> if kh == maxBound then kh else succ kh
                  _                           -> kh
      in (kh', kh')

  -- Select row for current pass/head/row; latch on 'clear' so it stays stable
  rowSel :: Signal System (RowI8E ModelDimension)
  rowSel = mux (passReg .==. pure PassQ)
                (selectRowQ (map wqHeadQ headsQv) <$> qHeadReg <*> rIdxS)
        $ mux (passReg .==. pure PassK)
                (selectRowK (map wkHeadQ headsQv) <$> kvHeadReg <*> rIdxS)
                (selectRowV (map wvHeadQ headsQv) <$> kvHeadReg <*> rIdxS)

  rowLat :: Signal System (RowI8E ModelDimension)
  rowLat = regEn (repeat 0, 0) clearS rowSel

  -- Stream x by selecting current column
  xNow :: Signal System FixedPoint
  xNow = (!!) <$> xHat <*> colIdxS

  -- One-row kernel
  (yRow, yDone) = matVecRowSeq clearS enS lastCol rowLat xNow

  -- Per-head work buffer (collects rows for the current head only)
  workBuf :: Signal System (Vec HeadDimension FixedPoint)
  workBuf = mealy goWork (repeat 0) (bundle (rowDone, rIdxS, yRow, clearS))
   where
    goWork wb (rd, r, y, cl) =
      let wb0 = if cl then repeat 0 else wb
          wb1 = if rd then replace r y wb0 else wb0
      in (wb1, wb1)

  -- Final per-layer registers for Q/K/V; commit at head completion
  qReg :: Signal System (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
  qReg = mealy goQR (repeat (repeat 0)) (bundle (passReg, rowDone, rIdxS, qHeadReg, workBuf, seqPos))
   where
    goQR acc (p, rd, r, qh, wb, sp) =
      let headDone = (p == PassQ) && rd && r == maxBound
          wbRot    = applyRotation (rotaryQ (headsQv !! qh)) sp wb
          acc'     = if headDone then replace qh wbRot acc else acc
      in (acc', acc')
  qOut = qReg

  kReg :: Signal System (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  kReg = mealy goKR (repeat (repeat 0)) (bundle (passReg, rowDone, rIdxS, kvHeadReg, workBuf, seqPos))
   where
    goKR acc (p, rd, r, kh, wb, sp) =
      let headDone = (p == PassK) && rd && r == maxBound
          qh       = queryHeadIndex0 kh
          wbRot    = applyRotation (rotaryQ (headsQv !! qh)) sp wb
          acc'     = if headDone then replace kh wbRot acc else acc
      in (acc', acc')
  kOut = kReg

  vReg :: Signal System (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
  vReg = mealy goVR (repeat (repeat 0)) (bundle (passReg, rowDone, rIdxS, kvHeadReg, workBuf))
   where
    goVR acc (p, rd, r, kh, wb) =
      let headDone = (p == PassV) && rd && r == maxBound
          acc'     = if headDone then replace kh wb acc else acc
      in (acc', acc')
  vOut = vReg

  -- Stage-1 done when we leave PassV on the final KV head
  allDonePass :: Signal System Bool
  allDonePass =
    let headDoneV = (passReg .==. pure PassV) .&&. rowDone .&&. (rIdxS .==. pure maxBound) .&&. (kvHeadReg .==. pure maxBound)
        leaveV    = headDoneV  -- 1-cycle pulse
    in leaveV

-- Layer wrapper that runs only during Stage1 for the selected layer.
-- It normalizes x, runs the QKV engine, and when s1Done pulses, commits Q/K/V
-- into LayerData.
stage1ProjectQKVSeqLayer
  :: HiddenClockResetEnable System
  => MultiHeadAttentionComponentQ
  -> Index NumLayers
  -> Signal System ProcessingState
  -> Signal System LayerData
  -> ( Signal System LayerData    -- updated LayerData (Q,K,V committed on s1Done)
     , Signal System Bool )       -- s1Done pulse
stage1ProjectQKVSeqLayer mha layerIx psSig ldSig = (ldNext, s1Done)
 where
  isThisS1 =
    liftA2 (\ps _ -> processingStage ps == Stage1_ProjectQKV
                  && processingLayer ps == layerIx)
           psSig (pure ())

  -- start pulse on S1 entry for this layer
  isThisS1Prev = register False isThisS1
  startS1      = isThisS1 .&&. not <$> isThisS1Prev

  -- normalized input (FixedPoint RMS)
  xHat = rmsNormFwFix <$> (inputVector <$> ldSig) <*> pure (rmsAttF mha)

  (qS, kS, vS, doneS) =
    projectQKVSeq mha (sequencePosition <$> psSig) xHat startS1

  -- commit on done pulse; otherwise pass-through
  commit ld q k v True  = ld { queryVectors = q, keyVectors = k, valueVectors = v }
  commit ld _ _ _ False = ld

  ldNext = liftA5 commit <$> ldSig <*> qS <*> kS <*> vS <*> doneS
  s1Done = doneS
```

Why it works
- The single engine walks Q → K → V, one head at a time, one row at a time. Row data is latched on clear and the input element is taken from xHat !! colIdx, keeping perfect alignment with the kernel’s internal column counter. Results are accumulated per row into workBuf; when the last row of that head completes, we commit into the head’s slot. Rotary is applied only once per head on the fully assembled vector. We emit s1Done as a 1-cycle pulse at the end of the final V head’s last row.

FFN sequential engine (W1, W3, W2) using same kernel
File: project/llama2/Model/Layers/FeedForward/FFNSequential.hs
```haskell
{-# LANGUAGE ScopedTypeVariables #-}
module Model.Layers.FeedForward.FFNSequential
  ( runFFNSeq
  ) where

import Clash.Prelude
import Model.Config (ModelDimension, HiddenDimension)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D(..), RowI8E)
import Model.Helpers.MatVecI8E_Seq (matVecRowSeq)
import Model.Helpers.Seq.RowColCtrl (rowColCtrl, RowColCtrlOut(..))
import Model.Helpers.FixedPoint (rmsNormFwFix)
import Model.Layers.FeedForward.FeedForwardNetwork.Internal (sigmoidLinearUnitF)
import Model.Layers.Components.Quantized (FeedForwardNetworkComponentQ(..))

data Pass = PIdle | PW1 | PW3 | PW2 deriving (Generic, NFDataX, Eq, Show)

-- Sequential FFN with single engine:
-- 1) xHat = RMSNorm(x, fRMSFfnF)
-- 2) gate = SiLU(W1 * xHat)   [HiddenDimension]
-- 3) up   =       W3 * xHat   [HiddenDimension]
-- 4) y    = W2 * (gate * up)  [ModelDimension]
-- start is a 1-cycle pulse; done pulses after final W2 row.
runFFNSeq
  :: HiddenClockResetEnable System
  => FeedForwardNetworkComponentQ
  -> Signal System (Vec ModelDimension FixedPoint)   -- ^ x (pre-attention residual)
  -> Signal System Bool                              -- ^ start pulse (Stage4 entry)
  -> ( Signal System (Vec ModelDimension FixedPoint) -- ^ y
     , Signal System Bool )                          -- ^ done pulse
runFFNSeq ffn x start = (yReg, allDone)
 where
  xHat = rmsNormFwFix <$> x <*> pure (fRMSFfnF ffn)

  -- Common row controller; re-used for PW1/PW3 (rows=HiddenDim, cols=ModelDim) and PW2 (rows=ModelDim, cols=HiddenDim)
  rcW1W3 :: RowColCtrlOut HiddenDimension ModelDimension
  rcW1W3 = rowColCtrl start

  rcW2 :: RowColCtrlOut ModelDimension HiddenDimension
  rcW2  = rowColCtrl (allDoneW3)  -- start W2 right after W3 completes

  -- Pass FSM
  passReg :: Signal System Pass
  passReg = mealy go PIdle (bundle (start, allDoneW1, allDoneW3, allDoneW2))
   where
    go p (st, d1, d3, d2) =
      let p' = case p of
                 PIdle -> if st then PW1 else PIdle
                 PW1   -> if d1 then PW3 else PW1
                 PW3   -> if d3 then PW2 else PW3
                 PW2   -> if d2 then PIdle else PW2
      in (p', p')

  -- W1 mat-vec: accumulate HiddenDimension rows into gateBuf, apply SiLU per row
  rowW1 = (!!) (unQ2D (fW1Q ffn)) <$> (rowIdx rcW1W3)
  xColW1 = (!!) <$> xHat <*> (colIdx rcW1W3)
  (yRowW1, rowDoneW1) = matVecRowSeq (clear rcW1W3) (en rcW1W3) (lastCol rcW1W3) rowW1 xColW1

  gateBuf :: Signal System (Vec HiddenDimension FixedPoint)
  gateBuf = mealy goG (repeat 0) (bundle (rowDoneW1, rowIdx rcW1W3, yRowW1))
   where
    goG acc (rd, r, y) =
      let acc' = if rd then replace r (sigmoidLinearUnitF y) acc else acc
      in (acc', acc')
  allDoneW1 = allDone rcW1W3

  -- W3 mat-vec: accumulate HiddenDimension rows into upBuf
  rowW3 = (!!) (unQ2D (fW3Q ffn)) <$> (rowIdx rcW1W3)
  xColW3 = xColW1
  (yRowW3, rowDoneW3) = matVecRowSeq (clear rcW1W3) (en rcW1W3) (lastCol rcW1W3) rowW3 xColW3

  upBuf :: Signal System (Vec HiddenDimension FixedPoint)
  upBuf = mealy goU (repeat 0) (bundle (rowDoneW3, rowIdx rcW1W3, yRowW3))
   where
    goU acc (rd, r, y) =
      let acc' = if rd then replace r y acc else acc
      in (acc', acc')
  allDoneW3 = allDone rcW1W3

  -- W2 mat-vec over z = gate ⊙ up
  zBuf :: Signal System (Vec HiddenDimension FixedPoint)
  zBuf = zipWith (*) <$> gateBuf <*> upBuf

  rowW2 = (!!) (unQ2D (fW2Q ffn)) <$> (rowIdx rcW2)
  zCol  = (!!) <$> zBuf <*> (colIdx rcW2)
  (yRowW2, rowDoneW2) = matVecRowSeq (clear rcW2) (en rcW2) (lastCol rcW2) rowW2 zCol

  yReg :: Signal System (Vec ModelDimension FixedPoint)
  yReg = mealy goY (repeat 0) (bundle (rowDoneW2, rowIdx rcW2, yRowW2))
   where
    goY acc (rd, r, y) =
      let acc' = if rd then replace r y acc else acc
      in (acc', acc')

  allDoneW2 = allDone rcW2
  allDone   = allDoneW2
```

Why it works
- The same kernel is re-used three times. W1 and W3 share the same controller (same shape Hidden×Model), so we compute both in a single sweep (gate and up are independent), avoiding re-reading xHat. W2 starts automatically after W3 completes and uses its own controller because the matrix shape differs. We commit the final y after the last row. No floats, all FixedPoint math.

Minimal integration: feed s1Done into the pipeline
Replace your Stage-1 stub in Model.Core.Transformer to use the new per-layer driver. This change does not replace your Stage-1 combinational code yet; it only supplies a real s1Done pulse so sequencing becomes correct. You can decide when to switch LayerData updates over to the sequential version (the stage1ProjectQKVSeqLayer already computes updates on the s1Done pulse).

Edit: project/llama2/Model/Core/Transformer.hs
Replace the “Stage1 done handshake” stub with per-layer s1Done outputs and select the active layer’s flag, mirroring your write/attn selection.

```haskell
-- add at the top
import qualified Model.Layers.Attention.ProjectQKVSeq as QKVSeq (stage1ProjectQKVSeqLayer)

-- inside multiCycleTransformer, after 'transformerLayers' definition
-- Run the Stage-1 QKV sequencers for all layers in parallel; select current one
s1Results :: Vec NumLayers (Signal dom LayerData, Signal dom Bool)
s1Results =
  imap
    (\ix layerComp ->
       QKVSeq.stage1ProjectQKVSeqLayer
         (multiHeadAttention layerComp) ix processingState layerDataRegister)
    transformerLayers

s1DoneFlags :: Vec NumLayers (Signal dom Bool)
s1DoneFlags = map snd s1Results

-- Select this layer's s1Done for pipeline control
s1DoneThisLayer :: Signal dom Bool
s1DoneThisLayer = (!!) <$> sequenceA s1DoneFlags <*> layerIndex
```

Note
- This only feeds the done handshake today. To fully switch to the sequential data path, either:
  - Option A: in layerProcessor/stageProcessor, stop writing Q/K/V in Stage1 and let stage1ProjectQKVSeqLayer commit them on s1Done (as it already does).
  - Option B: merge stage1ProjectQKVSeqLayer’s LayerData commit into your existing layerDataSelector logic. Keep exactly one writer for LayerData in Stage1 to avoid write-after-write hazards.

Testing and synthesis tips
- Simulation:
  - Drive start pulses and verify that:
    - clear is 1 exactly one cycle at each new row,
    - en is asserted for exactly cols cycles,
    - lastCol goes high with en on the final column,
    - rowDone pulses one cycle after lastCol,
    - allDone pulses at the end of the last row of the last pass (V or W2).
  - Check that the work buffers match a software dot-product reference for a few rows.
- Tooling:
  - Clash: ghci to step Signals; add small QuickCheck-style checks for counter alignment.
  - Synthesis: Yosys/nextpnr or vendor tools; both row ROMs and parameter matrices should infer ROM/BRAM. If matrices are large, keep the explicit rom access pattern (row latched on clear) to nudge BRAM inference.
- Performance:
  - The kernel is rate-1 over columns (one MAC per cycle). You can push Fmax by registering rowSel (as shown), x (xNow), and the accumulator inside the kernel, and by constraining the multiplier to map to DSPs.
  - If you later add pipelining to matVecRowSeq, preserve the clear/en/lastCol handshake; the controllers above need no change.

Portability/safety
- All code avoids Float in datapaths; LUT contents are elaboration-time constants.
- No partial functions on Signals; bounds-safe indices via Vec (!!) with Index types.
- The FFN module reuses the single controller for W1/W3; if you prefer fully serialized (W1 then W3), instantiate two controllers or gate a single controller with an extra pass FSM.

Further reading
- Clash Manual (GHC/Clash types, BlockRAM/ROM inference).
- IEEE 1364-2005 Verilog LRM for downstream synthesis expectations (if you inspect generated Verilog).
- Hints on FixedPoint numerics and PoT scaling in your own Model.Numeric modules.

If you want, I can provide a small testbench module that compares the sequential engines against your current combinational matrixVectorMult for small dimensions (e.g., MODEL_260K).