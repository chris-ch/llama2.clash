module LLaMa2.Core.PipelineController
  ( PipelineOutputs(..)
  , runPipelineController
  -- NEW: Minimal controller
  , ControllerState(..)
  , runMinimalController
  ) where

import Clash.Prelude
import LLaMa2.Core.Types (ProcessingState(..), CycleStage(..))
import LLaMa2.Config (NumLayers, SequenceLength)

-- ============================================================================
-- OLD CONTROLLER (still in use)
-- ============================================================================

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
  -> PipelineOutputs dom
runPipelineController
  attnDoneThisLayer writeDoneThisLayer qkvValidThisLayer
  ffnDoneThisLayer classifierDone inputTokenValid = outs
 where
  stage4CanAdvance = mux ((processingLayer <$> procState) .==. pure maxBound)
                         (pure True)
                         (pure True)

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

  stageFinishedSig =
    mux (isStage Stage1_ProjectQKV)
        (mux atFirstStage1
             (inputTokenValid .&&. qkvValidThisLayer)
             (qkvValidThisLayer .&&. pure True)) $
    mux (isStage Stage2_WriteKV)
        (writeDoneThisLayer .&&. pure True) $
    mux (isStage Stage3_Attend)
        (attnDoneThisLayer .&&. pure True) $
    mux (isStage Stage4_FeedForward)
        (ffnDoneThisLayer .&&. stage4CanAdvance) $
    mux (isStage Stage5_Classifier)
        (classifierDone .&&. pure True) $
    mux (isStage Stage6_Bookkeeping) (pure True) $
    pure False

  stageCanAdvance =
    mux (isStage Stage1_ProjectQKV) (pure True) $
    mux (isStage Stage2_WriteKV)    (pure True) $
    mux (isStage Stage3_Attend)     (pure True) $
    mux (isStage Stage4_FeedForward) (pure True) $
    mux (isStage Stage5_Classifier)  (pure True) $
    pure True

  readyPulseRaw =
    let isClassifierDone = isStage Stage5_Classifier .&&. classifierDone .&&. pure True
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

-- ============================================================================
-- NEW MINIMAL CONTROLLER (being added)
-- ============================================================================

data ControllerState = ControllerState
  { currentLayer :: Index NumLayers
  , ctrlSeqPos   :: Index SequenceLength
  } deriving (Generic, NFDataX, Show, Eq)

initialControllerState :: ControllerState
initialControllerState = ControllerState
  { currentLayer = 0
  , ctrlSeqPos   = 0
  }

-- Minimal controller: only tracks layer and sequence position
runMinimalController ::
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ currentLayerDone
  -> Signal dom Bool  -- ^ inputTokenValid
  -> ( Signal dom (Index NumLayers)
     , Signal dom (Index SequenceLength)
     , Signal dom Bool  -- ^ readyForNewToken
     )
runMinimalController layerDone tokenValid = (layerIdx, posIdx, tokenReady)
 where
  state = register initialControllerState nextState

  isLastLayer = (currentLayer <$> state) .==. pure maxBound

  nextState = advance <$> state <*> layerDone <*> isLastLayer

  advance :: ControllerState -> Bool -> Bool -> ControllerState
  advance s done lastLayer
    | not done = s
    | not lastLayer = s { currentLayer = succ (currentLayer s) }
    | otherwise = ControllerState
        { currentLayer = 0
        , ctrlSeqPos = if ctrlSeqPos s == maxBound then 0 else succ (ctrlSeqPos s)
        }

  layerIdx = currentLayer <$> state
  posIdx = ctrlSeqPos <$> state
  tokenReady = (&&) <$> layerDone <*> isLastLayer
