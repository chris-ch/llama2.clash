module LLaMa2.Decoder.SequenceController
 (PipelineOutputs(..)
  , pipelineController
  , layerSequencer
  , ControllerState(..)
) where
import Clash.Prelude
import LLaMa2.Core.Types (ProcessingState (..), CycleStage (..))
import LLaMa2.Config (NumLayers, SequenceLength)

data PipelineOutputs dom = PipelineOutputs
  { processingState   :: Signal dom ProcessingState
  , stageSignal       :: Signal dom CycleStage
  , layerIndex        :: Signal dom (Index NumLayers)
  , seqPos            :: Signal dom (Index SequenceLength)
  , readyPulse        :: Signal dom Bool
  , stageFinished     :: Signal dom Bool
  }

data ControllerState = ControllerState
  { currentLayer :: Index NumLayers
  , ctrlSeqPos   :: Index SequenceLength
  } deriving (Generic, NFDataX, Show, Eq)

pipelineController ::
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ attnDoneThisLayer
  -> Signal dom Bool  -- ^ writeDoneThisLayer  
  -> Signal dom Bool  -- ^ qkvValidThisLayer
  -> Signal dom Bool  -- ^ ffnDoneThisLayer
  -> Signal dom Bool  -- ^ classifierDone
  -> Signal dom Bool  -- ^ inputTokenValid
  -> PipelineOutputs dom
pipelineController
  attnDoneThisLayer writeDoneThisLayer qkvValidThisLayer
  ffnDoneThisLayer classifierDone inputTokenValid = outs
 where
  -- Simplified: stageCanAdvance is always True, so advance when done
  procState = register initialProcessingState
    ((\s done -> if done then nextProcessingState s else s) <$> procState <*> stageFinishedSig)

  stageSig = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  isStage st = (== st) <$> stageSig

  isFirstStage :: ProcessingState -> Bool
  isFirstStage ps = processingStage ps == Stage1_ProjectQKV
                  && processingLayer ps == 0
                  && sequencePosition ps == 0

  atFirstStage1 = isFirstStage <$> procState

  -- Simplified: removed all ".&&. pure True" occurrences
  stageFinishedSig =
    mux (isStage Stage1_ProjectQKV)
        (mux atFirstStage1
             (inputTokenValid .&&. qkvValidThisLayer)
             qkvValidThisLayer) $
    mux (isStage Stage2_WriteKV)
        writeDoneThisLayer $
    mux (isStage Stage3_Attend)
        attnDoneThisLayer $
    mux (isStage Stage4_FeedForward)
        ffnDoneThisLayer $
    mux (isStage Stage5_Classifier)
        classifierDone $
    isStage Stage6_Bookkeeping

  -- Simplified: removed ".&&. pure True"
  readyPulseRaw =
    let isClassifierDone = isStage Stage5_Classifier .&&. classifierDone
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

initialControllerState :: ControllerState
initialControllerState = ControllerState
  { currentLayer = 0
  , ctrlSeqPos   = 0
  }

layerSequencer ::
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ currentLayerDone
  -> ( Signal dom (Index NumLayers)
     , Signal dom (Index SequenceLength)
     , Signal dom Bool  -- ^ readyForNewToken
     )
layerSequencer layerDone = (layerIdx, posIdx, tokenReady)
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
