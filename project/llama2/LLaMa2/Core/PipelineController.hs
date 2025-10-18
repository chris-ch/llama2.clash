module LLaMa2.Core.PipelineController
  ( PipelineOutputs(..)
  , runPipelineController
  ) where

import Clash.Prelude
import LLaMa2.Core.Types (ProcessingState(..), CycleStage(..))
import LLaMa2.Config (NumLayers, SequenceLength)

initialProcessingState :: ProcessingState
initialProcessingState = ProcessingState
  { processingStage  = Stage1_ProjectQKV
  , processingLayer  = 0
  , sequencePosition = 0
  }

nextProcessingState :: ProcessingState -> ProcessingState
nextProcessingState state = case processingStage state of
  Stage1_ProjectQKV -> state { processingStage = Stage2_WriteKV }
  Stage2_WriteKV    -> state { processingStage = Stage3_Attend }
  Stage3_Attend     -> state { processingStage = Stage4_FeedForward }
  Stage4_FeedForward ->
    if processingLayer state == maxBound
      then state { processingStage  = Stage5_Classifier }
      else state { processingStage  = Stage1_ProjectQKV
                 , processingLayer  = succ (processingLayer state)
                 }
  Stage5_Classifier -> state { processingStage = Stage6_Bookkeeping }
  Stage6_Bookkeeping ->
    state { processingStage  = Stage1_ProjectQKV
          , processingLayer  = 0
          , sequencePosition = if sequencePosition state == maxBound
                                then 0 else succ (sequencePosition state)
          }

data PipelineOutputs dom = PipelineOutputs
  { processingState   :: Signal dom ProcessingState
  , stageSignal       :: Signal dom CycleStage
  , layerIndex        :: Signal dom (Index NumLayers)
  , seqPos            :: Signal dom (Index SequenceLength)
  , readyPulse        :: Signal dom Bool
  , stageFinished     :: Signal dom Bool
  }

runPipelineController ::
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ attnDoneThisLayer
  -> Signal dom Bool  -- ^ writeDoneThisLayer  
  -> Signal dom Bool  -- ^ qkvValidThisLayer
  -> Signal dom Bool  -- ^ ffnDoneThisLayer
  -> Signal dom Bool  -- ^ classifierDone
  -> Signal dom Bool  -- ^ inputTokenValid
  -> Signal dom Bool  -- ^ downstreamReady (external consumer ready)
  -> PipelineOutputs dom
runPipelineController 
  attnDoneThisLayer writeDoneThisLayer qkvValidThisLayer 
  ffnDoneThisLayer classifierDone inputTokenValid downstreamReady = outs
 where
  advance s done = if done then nextProcessingState s else s
  procState = register initialProcessingState (advance <$> procState <*> stageFinishedSig)

  stageSig = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  -- readyPulse: one-cycle pulse when the last layer's FFN asserts done
  isStage st = (== st) <$> stageSig

  isFirstStage :: ProcessingState -> Bool
  isFirstStage ps = processingStage ps == Stage1_ProjectQKV
                  && processingLayer ps == 0
                  && sequencePosition ps == 0

  atFirstStage1 = isFirstStage <$> procState

  -- Stage completion depends on both internal completion AND downstream readiness
  stageFinishedSig =
    mux (isStage Stage1_ProjectQKV)
        (mux atFirstStage1
             (inputTokenValid .&&. qkvValidThisLayer)
             qkvValidThisLayer) $
    mux (isStage Stage2_WriteKV)    writeDoneThisLayer $
    mux (isStage Stage3_Attend)     attnDoneThisLayer $
    mux (isStage Stage4_FeedForward) ffnDoneThisLayer $
    mux (isStage Stage5_Classifier)  
        (classifierDone .&&. downstreamReady) $  -- FIX: Check downstream
    mux (isStage Stage6_Bookkeeping) (pure True) $
    pure False

  -- FIX: Ready pulse only when downstream accepts
  readyPulseRaw =
    let isClassifierDone = isStage Stage5_Classifier .&&. classifierDone .&&. downstreamReady
        rising now prev = now && not prev
    in rising <$> isClassifierDone <*> register False isClassifierDone

  outs = PipelineOutputs
      { processingState = procState
      , stageSignal     = stageSig
      , layerIndex      = layerIx
      , seqPos          = posIx
      , readyPulse      = readyPulseRaw
      , stageFinished   = stageFinishedSig
      }
