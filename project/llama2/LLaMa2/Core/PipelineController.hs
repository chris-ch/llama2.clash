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
  -- FIX: Add stage-to-stage readiness tracking
  stage2Ready = pure True  -- Stage2 (write) is always ready
  stage3Ready = pure True  -- Stage3 (attend) is always ready
  stage4Ready = pure True  -- Stage4 (FFN) has internal FSM
  
  -- Stage1 can advance when Stage2 is ready to accept
  stage1CanAdvance = stage2Ready
  
  -- Stage2 can advance when Stage3 is ready
  stage2CanAdvance = stage3Ready
  
  -- Stage3 can advance when Stage4 is ready  
  stage3CanAdvance = stage4Ready
  
  -- Stage4 can advance when next layer/classifier is ready
  stage4CanAdvance = mux ((processingLayer <$> procState) .==. pure maxBound)
                         (pure True)  -- Last layer -> classifier always ready for now
                         (pure True)  -- Next layer Stage1 always ready
  
  advance s done canAdvance = if done && canAdvance then nextProcessingState s else s
  procState = register initialProcessingState 
    (advance <$> procState <*> stageFinishedSig <*> stageCanAdvance)

  stageSig = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  isStage st = (== st) <$> stageSig

  isFirstStage :: ProcessingState -> Bool
  isFirstStage ps = processingStage ps == Stage1_ProjectQKV
                  && processingLayer ps == 0
                  && sequencePosition ps == 0

  atFirstStage1 = isFirstStage <$> procState

  -- Stage completion includes downstream readiness checks
  stageFinishedSig =
    mux (isStage Stage1_ProjectQKV)
        (mux atFirstStage1
             (inputTokenValid .&&. qkvValidThisLayer)
             (qkvValidThisLayer .&&. stage1CanAdvance)) $
    mux (isStage Stage2_WriteKV)    
        (writeDoneThisLayer .&&. stage2CanAdvance) $
    mux (isStage Stage3_Attend)     
        (attnDoneThisLayer .&&. stage3CanAdvance) $
    mux (isStage Stage4_FeedForward) 
        (ffnDoneThisLayer .&&. stage4CanAdvance) $
    mux (isStage Stage5_Classifier)  
        (classifierDone .&&. downstreamReady) $
    mux (isStage Stage6_Bookkeeping) (pure True) $
    pure False

  -- Determine if current stage can advance based on downstream
  stageCanAdvance =
    mux (isStage Stage1_ProjectQKV) stage1CanAdvance $
    mux (isStage Stage2_WriteKV)    stage2CanAdvance $
    mux (isStage Stage3_Attend)     stage3CanAdvance $
    mux (isStage Stage4_FeedForward) stage4CanAdvance $
    mux (isStage Stage5_Classifier)  downstreamReady $
    pure True

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
