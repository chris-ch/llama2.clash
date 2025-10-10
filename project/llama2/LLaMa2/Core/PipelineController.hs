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
      then state { processingStage  = Stage5_Bookkeeping }
      else state { processingStage  = Stage1_ProjectQKV
                 , processingLayer  = succ (processingLayer state)
                 }
  Stage5_Bookkeeping ->
    state { processingStage  = Stage1_ProjectQKV
          , processingLayer  = 0
          , sequencePosition =
              if sequencePosition state == maxBound
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

-- Now includes 's1DoneThisLayer'
runPipelineController
  :: HiddenClockResetEnable dom
  => Signal dom Bool     -- ^ attnDoneThisLayer (Stage3)
  -> Signal dom Bool     -- ^ writeDoneThisLayer (Stage2)
  -> Signal dom Bool     -- ^ s1DoneThisLayer (Stage1)
  -> Signal dom Bool     -- ^ inputTokenValid (only used at very first (L0,P0))
  -> PipelineOutputs dom
runPipelineController attnDoneThisLayer writeDoneThisLayer s1DoneThisLayer inputTokenValid = outs
 where
  advance s done = if done then nextProcessingState s else s
  procState = register initialProcessingState (advance <$> procState <*> stageFinishedSig)

  stageSig = processingStage <$> procState
  layerIx  = processingLayer <$> procState
  posIx    = sequencePosition <$> procState

  -- readyPulse = rising edge when entering last-layer FFN
  isLastLayerFFN =
    liftA2 (\ps _ -> processingStage ps == Stage4_FeedForward
                  && processingLayer ps == maxBound)
           procState (pure ())
  readyPulseRaw =
    let rising now prev = now && not prev
    in  liftA2 rising isLastLayerFFN (register False isLastLayerFFN)

  atFirstStage1 =
    liftA2 (\ps _ -> processingStage ps == Stage1_ProjectQKV
                  && processingLayer ps == 0
                  && sequencePosition ps == 0)
           procState (pure ())

  isStage st = (== st) <$> stageSig
  stageFinishedSig =
    mux (isStage Stage1_ProjectQKV)
         (mux atFirstStage1 inputTokenValid s1DoneThisLayer) $
    mux (isStage Stage2_WriteKV)     writeDoneThisLayer      $
    mux (isStage Stage3_Attend)      attnDoneThisLayer       $
    mux (isStage Stage4_FeedForward) (not <$> readyPulseRaw) $
    mux (isStage Stage5_Bookkeeping) (pure True)             $
    pure False

  outs = PipelineOutputs
    { processingState = procState
    , stageSignal     = stageSig
    , layerIndex      = layerIx
    , seqPos          = posIx
    , readyPulse      = readyPulseRaw
    , stageFinished   = stageFinishedSig
    }
  