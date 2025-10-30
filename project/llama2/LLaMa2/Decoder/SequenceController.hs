module LLaMa2.Decoder.SequenceController
 (
  PipelineOutputs(..)
  , pipelineController
  , sequenceController
  , SequenceState(..)
) where
import Clash.Prelude
import LLaMa2.Types.LayerData (ProcessingState (..), CycleStage (..))
import LLaMa2.Types.ModelConfig  (NumLayers, SequenceLength)

data PipelineOutputs dom = PipelineOutputs
  { processingState   :: Signal dom ProcessingState
  , stageSignal       :: Signal dom CycleStage
  , layerIndex        :: Signal dom (Index NumLayers)
  , seqPos            :: Signal dom (Index SequenceLength)
  , readyPulse        :: Signal dom Bool
  , stageFinished     :: Signal dom Bool
  }

data SequenceState = SequenceState
  { currentLayer :: Index NumLayers
  , seqPosition  :: Index SequenceLength
  } deriving (Generic, NFDataX, Show, Eq)

pipelineController ::
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ attnDoneThisLayer
  -> Signal dom Bool  -- ^ writeDoneThisLayer  
  -> Signal dom Bool  -- ^ qkvValidThisLayer
  -> Signal dom Bool  -- ^ ffnDoneThisLayer
  -> Signal dom Bool  -- ^ classifierDone
  -> PipelineOutputs dom
pipelineController
  attnDoneThisLayer writeDoneThisLayer qkvValidThisLayer
  ffnDoneThisLayer classifierDone = outs
 where
  procState = register initialProcessingState
    ((\s done -> if done then nextProcessingState s else s) <$> procState <*> stageFinished)

  stage = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  isStage st = (== st) <$> stage

  stageFinished =
    mux (isStage Stage1_ProjectQKV)
        qkvValidThisLayer $
    mux (isStage Stage2_WriteKV)
        writeDoneThisLayer $
    mux (isStage Stage3_Attend)
        attnDoneThisLayer $
    mux (isStage Stage4_FeedForward)
        ffnDoneThisLayer $
    mux (isStage Stage5_Classifier)
        classifierDone $
    pure False

  readyPulseRaw =
    let isClassifierDone = isStage Stage5_Classifier .&&. classifierDone
        rising now prev = now && not prev
    in rising <$> isClassifierDone <*> register False isClassifierDone

  outs = PipelineOutputs
      { processingState = procState
      , stageSignal     = stage
      , layerIndex      = layerIx
      , seqPos          = posIx
      , readyPulse      = readyPulseRaw
      , stageFinished   = stageFinished
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
  Stage5_Classifier -> state { processingStage  = Stage1_ProjectQKV
          , processingLayer  = 0
          , sequencePosition = if sequencePosition state == maxBound
                                then 0 else succ (sequencePosition state)
          }

initialControllerState :: SequenceState
initialControllerState = SequenceState
  { currentLayer = 0
  , seqPosition   = 0
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

  advance :: SequenceState -> Bool -> Bool -> SequenceState
  advance s done lastLayer
    | not done = s
    | not lastLayer = s { currentLayer = succ (currentLayer s) }
    | otherwise = SequenceState
        { currentLayer = 0
        , seqPosition = if seqPosition s == maxBound then 0 else succ (seqPosition s)
        }

  layerIdx = currentLayer <$> state
  posIdx = seqPosition <$> state
  tokenReady = (&&) <$> layerDone <*> isLastLayer

-- | Unified sequence controller with clean interface
-- Wraps layerSequencer to return SequenceState instead of tuple
sequenceController
  :: HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ layerComplete (FFN done from last layer)
  -> ( Signal dom SequenceState  -- ^ Current sequence state
     , Signal dom Bool            -- ^ readyPulse (token complete)
     )
sequenceController layerComplete =
  (state, readyPulse)
  where
    (layerIdx, seqPosIdx, readyPulse) = layerSequencer layerComplete
    state = SequenceState <$> layerIdx <*> seqPosIdx
