module LLaMa2.Decoder.SequenceController
 ( SequenceController(..)
 , sequenceController
 ) where

import Clash.Prelude
import LLaMa2.Types.LayerData (CycleStage (..))
import LLaMa2.Types.ModelConfig (NumLayers, SequenceLength)

-- | Unified controller state: stage, active layer index, sequence position.
data UnifiedState = UnifiedState
  { stage  :: CycleStage
  , layer  :: Index NumLayers
  , seqPos :: Index SequenceLength
  } deriving (Generic, NFDataX, Show, Eq)

-- | Controller outputs: everything the pipeline needs to observe.
data SequenceController dom = SequenceController
  { processingStage :: Signal dom CycleStage
  , currentLayer    :: Signal dom (Index NumLayers)
  , seqPosition     :: Signal dom (Index SequenceLength)
  , readyPulse      :: Signal dom Bool       -- rising pulse when token completes
  , layerValidIn    :: Signal dom Bool       -- true when we should feed "valid" to active layer
  }

-- | Centralized stage controller.
-- The controller receives "done" signals coming from the active layer logic:
-- qkvDone, writeDone, attnDone, ffnDone, classifierDone.
--
-- It *owns* the stage transitions, layer index increment, and seqPosition increment.
-- Other modules only observe the outputs; they don't control progression.
sequenceController :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ qkvDone
  -> Signal dom Bool  -- ^ writeDone
  -> Signal dom Bool  -- ^ attnDone
  -> Signal dom Bool  -- ^ ffnDone
  -> Signal dom Bool  -- ^ classifierDone
  -> SequenceController dom
sequenceController qkvDone writeDone attnDone ffnDone classifierDone =
  SequenceController
    { processingStage = stageS
    , currentLayer    = layerS
    , seqPosition     = seqPosS
    , readyPulse      = readyPulseS
    , layerValidIn    = stageS .==. pure Stage1_ProjectQKV
    }
  where
    -- initial (power-up) controller state
    initState :: UnifiedState
    initState = UnifiedState
      { stage  = Stage1_ProjectQKV
      , layer  = 0
      , seqPos = 0
      }

    -- register holds the controller state
    state :: Signal dom UnifiedState
    state = register initState nextState

    stageS   = stage <$> state
    layerS   = layer <$> state
    seqPosS  = seqPos <$> state

    -- choose which 'done' signal matters based on current stage
    stageDone :: Signal dom Bool
    stageDone =
      let is s = stageS .==. pure s
      in  mux (is Stage1_ProjectQKV) qkvDone $
          mux (is Stage2_WriteKV)     writeDone $
          mux (is Stage3_Attend)      attnDone $
          mux (is Stage4_FeedForward) ffnDone $
          mux (is Stage5_Classifier)  classifierDone $
          pure False

    -- compute next state only when the current stage is done; otherwise hold state
    nextState :: Signal dom UnifiedState
    nextState = mux stageDone (advance <$> state) state

    -- pure function: how to advance from a given unified state
    advance :: UnifiedState -> UnifiedState
    advance s = case stage s of
      Stage1_ProjectQKV  -> s { stage = Stage2_WriteKV }
      Stage2_WriteKV     -> s { stage = Stage3_Attend }
      Stage3_Attend      -> s { stage = Stage4_FeedForward }
      Stage4_FeedForward ->
        if layer s == maxBound
          then s { stage = Stage5_Classifier }
          else s { stage = Stage1_ProjectQKV, layer = succ (layer s) }
      Stage5_Classifier  ->
        let nextSeq = if seqPos s == maxBound then 0 else succ (seqPos s)
        in UnifiedState
             { stage  = Stage1_ProjectQKV
             , layer  = 0
             , seqPos = nextSeq
             }

    -- ready pulse: a rising pulse when we observe (stage == Classifier) && classifierDone
    -- (we detect the rising edge of that conjunction)
    isClassifierAndDone :: Signal dom Bool
    isClassifierAndDone = (stageS .==. pure Stage5_Classifier) .&&. classifierDone

    -- rising-edge detector: true only for one cycle when sig goes False->True
    risingEdge :: Signal dom Bool -> Signal dom Bool
    risingEdge sig = (&&) <$> sig <*> (not <$> register False sig)

    readyPulseS = risingEdge isClassifierAndDone
