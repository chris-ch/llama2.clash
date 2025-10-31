module LLaMa2.Decoder.SimplifiedSequenceController
 ( UnifiedController(..)
 , unifiedController
 , UnifiedState(..)
 ) where

import Clash.Prelude
import LLaMa2.Types.LayerData (ProcessingState (..), CycleStage (..))
import LLaMa2.Types.ModelConfig (NumLayers, SequenceLength)

-- | State combining stage, layer, and sequence position
data UnifiedState = UnifiedState
  { stage    :: CycleStage
  , layer    :: Index NumLayers
  , seqPos   :: Index SequenceLength
  } deriving (Generic, NFDataX, Show, Eq)

-- | All controller outputs in one record
data UnifiedController dom = UnifiedController
  { -- Legacy outputs (keep for backward compatibility during transition)
    processingState :: Signal dom ProcessingState
  , currentLayer    :: Signal dom (Index NumLayers)
  , seqPosition     :: Signal dom (Index SequenceLength)
  , readyPulse      :: Signal dom Bool
  , stageComplete   :: Signal dom Bool
  
  -- Centralized enable signals (one per stage)
  , enableQKV        :: Signal dom Bool  -- Stage1_ProjectQKV is active
  , enableWriteKV    :: Signal dom Bool  -- Stage2_WriteKV is active
  , enableAttend     :: Signal dom Bool  -- Stage3_Attend is active
  , enableFFN        :: Signal dom Bool  -- Stage4_FeedForward is active
  , enableClassifier :: Signal dom Bool  -- Stage5_Classifier is active
  }

-- | Single controller
unifiedController ::
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ qkvDone
  -> Signal dom Bool  -- ^ writeDone
  -> Signal dom Bool  -- ^ attnDone
  -> Signal dom Bool  -- ^ ffnDone
  -> Signal dom Bool  -- ^ classifierDone
  -> UnifiedController dom
unifiedController qkvDone writeDone attnDone ffnDone classifierDone = 
  UnifiedController
    { processingState = toProcessingState <$> state
    , currentLayer    = layer <$> state
    , seqPosition     = seqPos <$> state
    , readyPulse      = readyPulse'
    , stageComplete   = stageDone
    
    -- Generate enable signals directly from current stage
    , enableQKV        = (stage <$> state) .==. pure Stage1_ProjectQKV
    , enableWriteKV    = (stage <$> state) .==. pure Stage2_WriteKV
    , enableAttend     = (stage <$> state) .==. pure Stage3_Attend
    , enableFFN        = (stage <$> state) .==. pure Stage4_FeedForward
    , enableClassifier = (stage <$> state) .==. pure Stage5_Classifier
    }
  where
    initialState = UnifiedState
      { stage  = Stage1_ProjectQKV
      , layer  = 0
      , seqPos = 0
      }
    
    state = register initialState nextState

    -- Determine if current stage is complete based on stage type
    stageDone = mux ((stage <$> state) .==. pure Stage1_ProjectQKV) qkvDone $
                mux ((stage <$> state) .==. pure Stage2_WriteKV)    writeDone $
                mux ((stage <$> state) .==. pure Stage3_Attend)     attnDone $
                mux ((stage <$> state) .==. pure Stage4_FeedForward) ffnDone $
                mux ((stage <$> state) .==. pure Stage5_Classifier) classifierDone $
                pure False

    -- Advance state when stage completes
    nextState = mux stageDone (advance <$> state) state

    -- State advancement logic
    advance :: UnifiedState -> UnifiedState
    advance s = case stage s of
      Stage1_ProjectQKV  -> s { stage = Stage2_WriteKV }
      Stage2_WriteKV     -> s { stage = Stage3_Attend }
      Stage3_Attend      -> s { stage = Stage4_FeedForward }
      Stage4_FeedForward -> 
        if layer s == maxBound
          then s { stage = Stage5_Classifier }
          else s { stage = Stage1_ProjectQKV, layer = succ (layer s) }
      Stage5_Classifier  -> s 
        { stage  = Stage1_ProjectQKV
        , layer  = 0
        , seqPos = if seqPos s == maxBound then 0 else succ (seqPos s)
        }

    -- Token complete: rising edge of (Stage5_Classifier && classifierDone)
    isTokenComplete = ((stage <$> state) .==. pure Stage5_Classifier) .&&. classifierDone
    readyPulse' = risingEdge isTokenComplete
    
    risingEdge sig = (&&) <$> sig <*> (not <$> register False sig)

-- | Convert unified state to ProcessingState for compatibility
toProcessingState :: UnifiedState -> ProcessingState
toProcessingState s = ProcessingState
  { processingStage  = stage s
  , processingLayer  = layer s
  , sequencePosition = seqPos s
  }
