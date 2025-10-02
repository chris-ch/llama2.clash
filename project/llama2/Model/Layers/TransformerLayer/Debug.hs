module Model.Layers.TransformerLayer.Debug (
    multiCycleTransformerLayerDbg
) where

import Clash.Prelude
import Model.Core.Types (ProcessingState(..), LayerData(..), CycleStage(..))
import Model.Config
  ( ModelDimension
  , NumLayers, NumQueryHeads, NumKeyValueHeads
  , HeadDimension, SequenceLength
  )
import qualified Model.Memory.KVCacheBank as Cache

import Model.Layers.TransformerLayer (TransformerLayerComponent(..))

import qualified Model.Layers.Attention.MultiHeadAttention as MultiHeadAttention (projectQKV)
import Model.Layers.Components.Quantized
  ( FeedForwardNetworkComponentQ(..)
  , MultiHeadAttentionComponentQ(..)
  , EmbeddingComponentQ(..)
  )

import qualified Model.Layers.FeedForward.FeedForwardNetwork as FeedForwardNetwork (computeFeedForward)
import Model.Helpers.MatVecI8E (matrixVectorMult)
import Model.Numeric.Types (FixedPoint)
import Helpers (liftA4)
import Model.Layers.Attention.AttentionHead (attendHead)
import Model.Memory.KVCacheBank.RowStreamer (kvRowStreamer)
import Model.Layers.Attention.AttendSequential (attendHeadSeq)
import Model.Config.Debug (AttnMode(..), attnMode, attnEps)
import Model.Memory.RamOps (toRamOperation)

-- | Multi-cycle transformer layer with debug hooks for shadow BRAM and row error checks
multiCycleTransformerLayerDbg
  :: HiddenClockResetEnable dom
  => TransformerLayerComponent
  -> Cache.KVRamOwner dom
  -> Index NumLayers
  -> Signal dom ProcessingState
  -> Signal dom LayerData
  -> ( Signal dom LayerData
     , Signal dom Bool
     , Signal dom Bool
     , Signal dom LayerData
     , Signal dom Bool
     , Signal dom Bool
     )
multiCycleTransformerLayerDbg layer kvOwner layerIx psSig idSig =
  ( nextLayerData
  , writeDone
  , attentionDone
  , commitStage3
  , kErrHold
  , vErrHold
  )
 where
  mhaQ = multiHeadAttention layer
  ffnQ = feedforwardNetwork layer

  -- Initialize accumulators
  initHeadOuts  = repeat (pure (repeat 0))
  initHeadDone  = repeat (pure False)
  initWriteDone = repeat (pure False)
  initKErr      = pure False
  initVErr      = pure False

  -- Fold over KV banks
  (perHeadOutputs, perHeadDone, perBankWrites, kErrOR, vErrOR) =
    foldl (fillOneBankDbg layerIx psSig kvOwner idSig)
          (initHeadOuts, initHeadDone, initWriteDone, initKErr, initVErr)
          indicesI

  -- Attention done pulse
  allHeadsDone     = and <$> sequenceA perHeadDone
  allHeadsDonePrev = register False allHeadsDone
  attentionDone    = liftA2 (\now prev -> now && not prev) allHeadsDone allHeadsDonePrev

  -- Stage processor
  baseNextLayerData = liftA2 (stageProcessor mhaQ ffnQ layerIx) psSig idSig

  -- Write-done signal
  writeDone = liftA2 (\ps banksDone -> processingStage ps == Stage2_WriteKV
                                        && processingLayer ps == layerIx
                                        && banksDone)
                      psSig
                      (and <$> sequenceA perBankWrites)

  -- Project WO per head and sum
  perHeadProjVec = zipWith (\woQ sig -> matrixVectorMult woQ <$> sig) (mWoQ mhaQ) perHeadOutputs
  perHeadProj    = sequenceA perHeadProjVec
  woHeads        = foldl1 (zipWith (+)) <$> perHeadProj

  -- Combine input + attention output
  xAfterAttn = liftA2 (zipWith (+) . inputVector) idSig woHeads

  -- Next layer data
  nextLayerData = liftA4 (\ps cur att done ->
                            if processingLayer ps == layerIx && processingStage ps == Stage3_Attend && done
                               then cur { attentionOutput = att }
                               else cur)
                         psSig
                         baseNextLayerData
                         xAfterAttn
                         attentionDone

  -- Commit signal (same as nextLayerData but from input directly)
  commitStage3 = liftA4 (\ps cur att done ->
                           if processingLayer ps == layerIx && processingStage ps == Stage3_Attend && done
                              then cur { attentionOutput = att }
                              else cur)
                        psSig
                        idSig
                        xAfterAttn
                        attentionDone

  -- Hold OR of KV row errors until attention done
  kErrHold = regEn False attentionDone kErrOR
  vErrHold = regEn False attentionDone vErrOR

-- | Per-layer stage processing
stageProcessor
  :: MultiHeadAttentionComponentQ
  -> FeedForwardNetworkComponentQ
  -> Index NumLayers
  -> ProcessingState
  -> LayerData
  -> LayerData
stageProcessor mhaQ ffnQ layerIx ps idata
  | processingLayer ps /= layerIx = idata
  | otherwise = case processingStage ps of
      Stage1_ProjectQKV ->
        let (qs, ks, vs) = MultiHeadAttention.projectQKV mhaQ (sequencePosition ps) (inputVector idata)
        in idata { queryVectors = qs, keyVectors = ks, valueVectors = vs }
      Stage2_WriteKV     -> idata
      Stage3_Attend      -> idata
      Stage4_FeedForward ->
        let ffnOut = FeedForwardNetwork.computeFeedForward ffnQ (attentionOutput idata)
        in idata { feedForwardOutput = ffnOut }
      Stage5_Bookkeeping -> idata

fillOneBankDbg
  :: HiddenClockResetEnable dom
  => Index NumLayers
  -> Signal dom ProcessingState
  -> Cache.KVRamOwner dom
  -> Signal dom LayerData
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool)
     , Signal dom Bool
     , Signal dom Bool )
  -> Index NumKeyValueHeads
  -> ( Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
     , Vec NumQueryHeads (Signal dom Bool)
     , Vec NumKeyValueHeads (Signal dom Bool)
     , Signal dom Bool
     , Signal dom Bool )
fillOneBankDbg layerIx psSig kvOwner idSig
               (headOutAcc, headDoneAcc, writeDoneAcc, kErrAcc, vErrAcc) kvIx =
  let
    -- Stage predicates
    isStage3 = liftA2 (\ps _ -> processingStage ps == Stage3_Attend && processingLayer ps == layerIx)
                      psSig
                      (pure ())
    isStage2 = liftA2 (\ps _ -> processingStage ps == Stage2_WriteKV && processingLayer ps == layerIx)
                      psSig
                      (pure ())

    -- Stage3 pulse
    attnPrev = register False isStage3
    clearS3  = liftA2 (\now prev -> now && not prev) isStage3 attnPrev

    -- Seq position
    seqPos = sequencePosition <$> psSig

    -- Query head mapping
    qIdx0 :: Index NumQueryHeads
    qIdx0 = toEnum
        (min
            (natToNum @NumQueryHeads - 1)
            (fromEnum kvIx * (natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads))
        )

    qIdx1 :: Index NumQueryHeads
    qIdx1 = if hasQ1 then succ qIdx0 else qIdx0

    hasQ1 = (natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads) >= 2
             && (fromEnum kvIx * (natToNum @NumQueryHeads `div` natToNum @NumKeyValueHeads) + 1 <= natToNum @NumQueryHeads - 1)

    query0 = (\d -> queryVectors d !! qIdx0) <$> idSig
    query1 = if hasQ1 then (\d -> queryVectors d !! qIdx1) <$> idSig else pure (repeat 0)

    keyVec   = (\d -> keyVectors d !! kvIx) <$> idSig
    valueVec = (\d -> valueVectors d !! kvIx) <$> idSig

    -- Stage2: quantized write sequencer
    (wrAddr, kMant, kExp, vMant, vExp, writeDoneThisBank) =
      Cache.writeSequencer isStage2 seqPos (bundle (keyVec, valueVec))

    wrAddrS = mux isStage2 wrAddr (pure 0)
    kMantWr = mux isStage2 kMant (pure Nothing)
    vMantWr = mux isStage2 vMant (pure Nothing)
    kExpWr  = mux isStage2 kExp  (pure Nothing)
    vExpWr  = mux isStage2 vExp  (pure Nothing)

    -- Stage3: progressive row sequencing for unquantized BRAM
    (tAddrRow, stepEnRow, lastTRow) = attentionRowSequencer clearS3 isStage3 seqPos

    -- Unquantized BRAM ports
    wrKVRowK = mux isStage2 (Just <$> bundle (seqPos, keyVec)) (pure Nothing)
    wrKVRowV = mux isStage2 (Just <$> bundle (seqPos, valueVec)) (pure Nothing)
    bank     = Cache.kvBanks kvOwner !! kvIx
    (kRowF_A, _kRowF_B) = trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                                                 (toRamOperation seqPos wrKVRowK)
    (vRowF_A, _vRowF_B) = trueDualPortBlockRam (toRamOperation tAddrRow (pure Nothing))
                                                 (toRamOperation seqPos wrKVRowV)

    -- Baseline attend (combinational from register mirror)
    kvKeysAll = mealy (\mem (we,p,row) -> let mem' = if we then replace p row mem else mem in (mem', mem'))
                      (repeat (repeat 0))
                      (bundle (isStage2, seqPos, keyVec))
    kvValsAll = mealy (\mem (we,p,row) -> let mem' = if we then replace p row mem else mem in (mem', mem'))
                      (repeat (repeat 0))
                      (bundle (isStage2, seqPos, valueVec))
    out0_baseline = liftA4 attendHead query0 kvKeysAll kvValsAll seqPos
    out1_baseline = if hasQ1 then liftA4 attendHead query1 kvKeysAll kvValsAll seqPos else pure (repeat 0)
    doneBaseline  = clearS3

    -- Streamed (shadow) attend
    (kRowQ, vRowQ, rowValidQ, lastTQ) = kvRowStreamer bank clearS3 isStage3 seqPos wrAddrS kMantWr kExpWr vMantWr vExpWr
    (out0_seqQ, done0_seqQ) = attendHeadSeq clearS3 rowValidQ query0 kRowQ vRowQ lastTQ
    (out1_seqQ, done1_seqQ) = if hasQ1 then attendHeadSeq clearS3 rowValidQ query1 kRowQ vRowQ lastTQ else (pure (repeat 0), pure False)

    -- Streamed unquantized
    (out0_seqF, done0_seqF) = attendHeadSeq clearS3 stepEnRow query0 kRowF_A vRowF_A lastTRow
    (out1_seqF, done1_seqF) = if hasQ1 then attendHeadSeq clearS3 stepEnRow query1 kRowF_A vRowF_A lastTRow else (pure (repeat 0), pure False)

    -- Selection
    (out0_sel, out1_sel, done0_sel, done1_sel) =
      case attnMode of
        AttnBaseline      -> (out0_baseline, out1_baseline, doneBaseline, doneBaseline)
        AttnShadowBRAM    -> (out0_baseline, out1_baseline, doneBaseline, doneBaseline)
        AttnReplaceBRAMF  -> (out0_seqF, out1_seqF, done0_seqF, done1_seqF)
        AttnReplaceBRAMQ  -> (out0_seqQ, out1_seqQ, done0_seqQ, done1_seqQ)

    -- Row error detection
    maxAbs v = foldl max 0 (map abs v)
    diffTooBig a b = maxAbs (zipWith (-) a b) > attnEps
    kRowErr     = diffTooBig <$> kRowQ <*> kRowF_A
    vRowErr     = diffTooBig <$> vRowQ <*> vRowF_A
    lastKRowErr = regEn False lastTQ kRowErr
    lastVRowErr = regEn False lastTQ vRowErr

    -- Accumulate outputs
    headOutAcc0  = replace qIdx0 out0_sel headOutAcc
    headOutAcc1  = if hasQ1 then replace qIdx1 out1_sel headOutAcc0 else headOutAcc0
    headDoneAcc0 = replace qIdx0 done0_sel headDoneAcc
    headDoneAcc1 = if hasQ1 then replace qIdx1 done1_sel headDoneAcc0 else headDoneAcc0
    writeDoneAcc1 = replace kvIx writeDoneThisBank writeDoneAcc
    kErrAcc1 = liftA2 (||) kErrAcc lastKRowErr
    vErrAcc1 = liftA2 (||) vErrAcc lastVRowErr

  in (headOutAcc1, headDoneAcc1, writeDoneAcc1, kErrAcc1, vErrAcc1)

-- | Generate row read addresses and step/last signals for progressive replacement
-- Unquantized row BRAMs (FixedPoint) used only for progressive replacement:
-- Two true-dual-port BRAMs with (depth = SequenceLength, payload = Vec HeadDimension FixedPoint)
-- Port A: Stage3 reads; Port B: Stage2 writes current row at (seqPos)
-- Address/Write streams for BRAM-F
-- | Generate row read addresses and step/last signals for progressive replacement
attentionRowSequencer ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool                  -- ^ clearS3 (stage 3 entry pulse)
  -> Signal dom Bool                  -- ^ isStage3Attention
  -> Signal dom (Index SequenceLength) -- ^ seqPosSignal
  -> ( Signal dom (Index SequenceLength)  -- tAddrRow
     , Signal dom Bool                   -- stepEnRow
     , Signal dom Bool )                 -- lastTRow
attentionRowSequencer clearS3 isStage3Attention seqPosSignal =
  let
    -- === Step 1: manage row counter ===
    -- Reset to 0 on Stage3 entry
    rowCounter :: Signal dom (Index SequenceLength)
    rowCounter = mealy rowCounterT 0 (bundle (clearS3, isStage3Attention, seqPosSignal))

    rowCounterT :: Index SequenceLength -> (Bool, Bool, Index SequenceLength)
                -> (Index SequenceLength, Index SequenceLength)
    rowCounterT t (clearPulse, stageActive, pos) =
      let
        tStart = if clearPulse then 0 else t
        step   = stageActive
        last   = step && tStart == pos
        tNext  = if not step || last then tStart else succ tStart
      in (tNext, tStart)

    -- === Step 2: detect step enable ===
    stepNow :: Signal dom Bool
    stepNow = liftA2 const isStage3Attention rowCounter

    stepEnRow :: Signal dom Bool
    stepEnRow = register False stepNow

    -- === Step 3: detect last row ===
    lastNow :: Signal dom Bool
    lastNow = liftA2 (==) rowCounter seqPosSignal

    lastTRow :: Signal dom Bool
    lastTRow = register False lastNow

  in (rowCounter, stepEnRow, lastTRow)
