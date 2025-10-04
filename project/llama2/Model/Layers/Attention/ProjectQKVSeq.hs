module Model.Layers.Attention.ProjectQKVSeq
  ( projectQKVSeq,
    stage1ProjectQKVSeqLayer,
  )
where

import Clash.Prelude
import Model.Config
  ( HeadDimension,
    ModelDimension,
    NumKeyValueHeads,
    NumLayers,
    NumQueryHeads,
    SequenceLength,
  )
import Model.Core.Types (CycleStage (..), LayerData (..), ProcessingState (..))
import Model.Helpers (liftA5)
import Model.Helpers.FixedPoint (rmsNormFwFix)
import Model.Helpers.MatVecI8E_Seq (matVecRowSeq)
import Model.Helpers.Seq.RowColCtrl (RowColCtrlOut (..), rowColCtrl)
import Model.Layers.Attention.MultiHeadAttention.Internal (applyRotation)
import Model.Layers.Components.Quantized (MultiHeadAttentionComponentQ (..), SingleHeadComponentQ (..))
import Model.Numeric.ParamPack (QArray2D (..), RowI8E)
import Model.Numeric.Types (FixedPoint)

-- Map each KV head to the query head it shares rotary/weight bank with
queryHeadIndex0 :: Index NumKeyValueHeads -> Index NumQueryHeads
queryHeadIndex0 kvIx =
  let nQ = natToNum @NumQueryHeads
      nKV = natToNum @NumKeyValueHeads
      base = min (nQ - 1) (fromEnum kvIx * (nQ `div` nKV))
   in toEnum base

-- Select current row for the active pass/head/row
selectRowQ :: Vec NumQueryHeads SingleHeadComponentQ -> Index NumQueryHeads -> Index HeadDimension -> RowI8E ModelDimension
selectRowQ headsQ qIx rIx =
  let QArray2D m = wqHeadQ (headsQ !! qIx)
  in m !! rIx

selectRowK :: Vec NumQueryHeads SingleHeadComponentQ -> Index NumKeyValueHeads -> Index HeadDimension -> RowI8E ModelDimension
selectRowK headsQ kvIx rIx =
  let qIx = queryHeadIndex0 kvIx
      QArray2D m = wkHeadQ (headsQ !! qIx)
  in m !! rIx

selectRowV :: Vec NumQueryHeads SingleHeadComponentQ -> Index NumKeyValueHeads -> Index HeadDimension -> RowI8E ModelDimension
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
projectQKVSeq ::
  forall dom.
  (HiddenClockResetEnable dom) =>
  MultiHeadAttentionComponentQ ->
  -- | seq pos for rotary
  Signal dom (Index SequenceLength) ->
  -- | input x (already RMS-normalized preferred)
  Signal dom (Vec ModelDimension FixedPoint) ->
  -- | start pulse
  Signal dom Bool ->
  ( Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint)), -- Q
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)), -- K
    Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint)), -- V
    Signal dom Bool -- done
  )
projectQKVSeq mha seqPos xHat start = (qOut, kOut, vOut, allDonePass)
  where
    headsQv = headsQ mha

    -- Controller per row (rows=headDim, cols=modelDim)
    rc :: RowColCtrlOut dom HeadDimension ModelDimension
    rc = rowColCtrl start
    clearS = clear rc
    enS = en rc
    lastCol' = lastCol rc
    colIdxS = colIdx rc
    rIdxS = rowIdx rc
    rowDone' = rowDone rc

    -- Pass/head iteration
    passReg :: Signal dom Pass
    passReg = mealy goPass PassIdle (bundle (start, rowDone', rIdxS, qHeadReg, kvHeadReg))
      where
        goPass p (st, rd, r, qh, kvh) =
          let onRowLast = rd && r == maxBound
              p' = case p of
                PassIdle -> if st then PassQ else PassIdle
                PassQ -> if onRowLast && qh == maxBound then PassK else PassQ
                PassK -> if onRowLast && kvh == maxBound then PassV else PassK
                PassV -> if onRowLast && kvh == maxBound then PassIdle else PassV
           in (p', p')

    -- head indices advance at head completion (last row done)
    qHeadReg :: Signal dom (Index NumQueryHeads)
    qHeadReg = mealy goQ 0 (bundle (passReg, rowDone', rIdxS))
      where
        goQ qh (p, rd, r) =
          let qh' = case p of
                PassQ | rd && r == maxBound -> if qh == maxBound then qh else succ qh
                _ -> qh
           in (qh', qh')

    kvHeadReg :: Signal dom (Index NumKeyValueHeads)
    kvHeadReg = mealy goKV 0 (bundle (passReg, rowDone', rIdxS))
      where
        goKV kh (p, rd, r) =
          let kh' = case p of
                PassK | rd && r == maxBound -> if kh == maxBound then kh else succ kh
                PassV | rd && r == maxBound -> if kh == maxBound then kh else succ kh
                _ -> kh
           in (kh', kh')

    -- Select row for current pass/head/row; latch on 'clear' so it stays stable
    rowSel :: Signal dom (RowI8E ModelDimension)
    rowSel = mux (passReg .==. pure PassQ)
                    (selectRowQ headsQv <$> qHeadReg <*> rIdxS)
            $ mux (passReg .==. pure PassK)
                    (selectRowK headsQv <$> kvHeadReg <*> rIdxS)
                    (selectRowV headsQv <$> kvHeadReg <*> rIdxS)

    rowLat :: Signal dom (RowI8E ModelDimension)
    rowLat = regEn (repeat 0, 0) clearS rowSel

    -- Stream x by selecting current column
    xNow :: Signal dom FixedPoint
    xNow = (!!) <$> xHat <*> colIdxS

    -- One-row kernel
    (yRow, _) = matVecRowSeq clearS enS lastCol' rowLat xNow

    -- Per-head work buffer (collects rows for the current head only)
    workBuf :: Signal dom (Vec HeadDimension FixedPoint)
    workBuf = mealy goWork (repeat 0) (bundle (rowDone', rIdxS, yRow, clearS))
      where
        goWork wb (rd, r, y, cl) =
          let wb0 = if cl then repeat 0 else wb
              wb1 = if rd then replace r y wb0 else wb0
           in (wb1, wb1)

    -- Final per-layer registers for Q/K/V; commit at head completion
    qReg :: Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
    qReg = mealy goQR (repeat (repeat 0)) (bundle (passReg, rowDone', rIdxS, qHeadReg, workBuf, seqPos))
      where
        goQR acc (p, rd, r, qh, wb, sp) =
          let headDone = (p == PassQ) && rd && r == maxBound
              wbRot = applyRotation (rotaryQ (headsQv !! qh)) sp wb
              acc' = if headDone then replace qh wbRot acc else acc
           in (acc', acc')
    qOut = qReg

    kReg :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
    kReg = mealy goKR (repeat (repeat 0)) (bundle (passReg, rowDone', rIdxS, kvHeadReg, workBuf, seqPos))
      where
        goKR acc (p, rd, r, kh, wb, sp) =
          let headDone = (p == PassK) && rd && r == maxBound
              qh = queryHeadIndex0 kh
              wbRot = applyRotation (rotaryQ (headsQv !! qh)) sp wb
              acc' = if headDone then replace kh wbRot acc else acc
           in (acc', acc')
    kOut = kReg

    vReg :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
    vReg = mealy goVR (repeat (repeat 0)) (bundle (passReg, rowDone', rIdxS, kvHeadReg, workBuf))
      where
        goVR acc (p, rd, r, kh, wb) =
          let headDone = (p == PassV) && rd && r == maxBound
              acc' = if headDone then replace kh wb acc else acc
           in (acc', acc')
    vOut = vReg

    -- Stage-1 done when we leave PassV on the final KV head
    allDonePass :: Signal dom Bool
    allDonePass =
      let headDoneV = (passReg .==. pure PassV) .&&. rowDone' .&&. (rIdxS .==. pure maxBound) .&&. (kvHeadReg .==. pure maxBound)
          leaveV = headDoneV -- 1-cycle pulse
       in leaveV

-- Layer wrapper that runs only during Stage1 for the selected layer.
-- It normalizes x, runs the QKV engine, and when s1Done pulses, commits Q/K/V
-- into LayerData.
stage1ProjectQKVSeqLayer ::
  (HiddenClockResetEnable dom) =>
  MultiHeadAttentionComponentQ ->
  Index NumLayers ->
  Signal dom ProcessingState ->
  Signal dom LayerData ->
  ( Signal dom LayerData, -- updated LayerData (Q,K,V committed on s1Done)
    Signal dom Bool -- s1Done pulse
  )
stage1ProjectQKVSeqLayer mha layerIx psSig ldSig = (ldNext, s1Done)
  where
    isThisS1 =
      liftA2
        ( \ps _ ->
            processingStage ps
              == Stage1_ProjectQKV
              && processingLayer ps
              == layerIx
        )
        psSig
        (pure ())

    -- start pulse on S1 entry for this layer
    isThisS1Prev = register False isThisS1
    startS1 = isThisS1 .&&. not <$> isThisS1Prev

    -- normalized input (FixedPoint RMS)
    xHat = rmsNormFwFix <$> (inputVector <$> ldSig) <*> pure (rmsAttF mha)

    (qS, kS, vS, doneS) =
      projectQKVSeq mha (sequencePosition <$> psSig) xHat startS1

    -- commit on done pulse; otherwise pass-through
    commit ld q k v True = ld {queryVectors = q, keyVectors = k, valueVectors = v}
    commit ld _ _ _ False = ld

    ldNext = liftA5 commit ldSig qS kS vS doneS
    s1Done = doneS
