module LLaMa2.Decoder.SequenceController
 ( SequenceController(..)
 , sequenceController
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
data SequenceController dom = SequenceController
  { -- Legacy outputs (keep for backward compatibility during transition)
    processingState :: Signal dom ProcessingState,
  currentLayer    :: Signal dom (Index NumLayers),
  seqPosition     :: Signal dom (Index SequenceLength),
  readyPulse      :: Signal dom Bool,
  stageComplete   :: Signal dom Bool,
  
  -- Simplified enable signals (one per major stage)
  enableAttention  :: Signal dom Bool,  -- Stage_Attention is active (handles QKV+Write+Attend internally)
  enableFFN        :: Signal dom Bool,  -- Stage_FeedForward is active
  enableClassifier :: Signal dom Bool   -- Stage_Classifier is active
  }

-- | Single controller with simplified stage management
sequenceController ::
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ attnDone (entire attention mechanism complete)
  -> Signal dom Bool  -- ^ ffnDone
  -> Signal dom Bool  -- ^ classifierDone
  -> SequenceController dom
sequenceController attnDone ffnDone classifierDone = 
  SequenceController
    { processingState = toProcessingState <$> state,
    currentLayer    = layer <$> state,
    seqPosition     = seqPos <$> state,
    readyPulse      = readyPulse',
    stageComplete   = stageDone
    
    -- Generate enable signals directly from current stage
    , enableAttention  = (stage <$> state) .==. pure Stage_Attention
    , enableFFN        = (stage <$> state) .==. pure Stage_FeedForward
    , enableClassifier = (stage <$> state) .==. pure Stage_Classifier
    }
  where
    initialState = UnifiedState
      { stage  = Stage_Attention
      , layer  = 0
      , seqPos = 0
      }
    
    state = register initialState nextState

    -- Determine if current stage is complete based on stage type
    stageDone = mux ((stage <$> state) .==. pure Stage_Attention)   attnDone $
                mux ((stage <$> state) .==. pure Stage_FeedForward) ffnDone $
                mux ((stage <$> state) .==. pure Stage_Classifier)  classifierDone $
                pure False

    -- Advance state when stage completes
    nextState = mux stageDone (advance <$> state) state

    -- State advancement logic (simplified)
    advance :: UnifiedState -> UnifiedState
    advance s = case stage s of
      Stage_Attention  -> s { stage = Stage_FeedForward }
      Stage_FeedForward -> 
        if layer s == maxBound
          then s { stage = Stage_Classifier }
          else s { stage = Stage_Attention, layer = succ (layer s) }
      Stage_Classifier  -> s 
        { stage  = Stage_Attention
        , layer  = 0
        , seqPos = if seqPos s == maxBound then 0 else succ (seqPos s)
        }

    -- Token complete: rising edge of (Stage_Classifier && classifierDone)
    isTokenComplete = ((stage <$> state) .==. pure Stage_Classifier) .&&. classifierDone
    readyPulse' = risingEdge isTokenComplete
    
    risingEdge sig = (&&) <$> sig <*> (not <$> register False sig)

-- | Convert unified state to ProcessingState for compatibility
toProcessingState :: UnifiedState -> ProcessingState
toProcessingState s = ProcessingState
  { processingStage  = stage s
  , processingLayer  = layer s
  , sequencePosition = seqPos s
  }
