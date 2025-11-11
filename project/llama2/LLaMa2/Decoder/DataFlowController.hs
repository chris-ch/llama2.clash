module LLaMa2.Decoder.DataFlowController
 ( DataFlowController(..)
 , dataFlowController
 , DataStage(..)
 ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, SequenceLength)

-- | High-level data flow state
data DataStage 
  = ProcessingLayer
  | Classifier
  deriving (Generic, NFDataX, Show, Eq)

-- | Controller state
data ControllerState = ControllerState
  { stage  :: DataStage
  , layer  :: Index NumLayers
  , seqPos :: Index SequenceLength
  } deriving (Generic, NFDataX, Show, Eq)

-- | Controller outputs
data DataFlowController dom = DataFlowController
  { processingStage :: Signal dom DataStage
  , currentLayer    :: Signal dom (Index NumLayers)
  , seqPosition     :: Signal dom (Index SequenceLength)
  , readyPulse      :: Signal dom Bool       -- token generation complete
  , layerValidIn    :: Signal dom Bool       -- pulse to start layer processing
  }

-- | Data flow controller - manages layer boundaries and token completion
dataFlowController :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool  -- ^ ffnDone: layer processing complete
  -> Signal dom Bool  -- ^ classifierDone: token classification complete
  -> DataFlowController dom
dataFlowController ffnDone classifierDone =
  DataFlowController
    { processingStage = stageS
    , currentLayer    = layerS
    , seqPosition     = seqPosS
    , readyPulse      = readyPulseS
    , layerValidIn    = layerValidInS
    }
  where
    -- Initial state
    initState :: ControllerState
    initState = ControllerState
      { stage  = ProcessingLayer
      , layer  = 0
      , seqPos = 0
      }

    -- State register
    state :: Signal dom ControllerState
    state = register initState nextState

    stageS   = stage <$> state
    layerS   = layer <$> state
    seqPosS  = seqPos <$> state

    -- Choose which done signal matters based on current stage
    controlDone :: Signal dom Bool
    controlDone = 
      mux (stageS .==. pure ProcessingLayer) ffnDone $
      mux (stageS .==. pure Classifier) classifierDone $
      pure False

    -- Advance only when control signals completion
    nextState :: Signal dom ControllerState
    nextState = mux controlDone (advance <$> state) state

    -- State machine: layer boundaries and token completion
    advance :: ControllerState -> ControllerState
    advance s = case stage s of
      ProcessingLayer ->
        if layer s == maxBound
          then s { stage = Classifier }           -- Last layer done, classify
          else s { layer = succ (layer s) }       -- Next layer
      
      Classifier ->
        -- Token complete, reset to layer 0 and increment sequence
        s { stage  = ProcessingLayer
          , layer  = 0
          , seqPos = if seqPos s == maxBound then 0 else succ (seqPos s)
          }

    -- Ready pulse when transitioning FROM Classifier
    readyPulseS :: Signal dom Bool
    readyPulseS = (stageS .==. pure Classifier) .&&. classifierDone

    -- Layer valid input: pulse when starting a new layer
    -- Pulses on: initial cycle, layer transitions, and returning from classifier
    firstCycle :: Signal dom Bool
    firstCycle = register True (pure False)
    
    layerValidInS :: Signal dom Bool
    layerValidInS = firstCycle .||. 
                    ((stageS .==. pure ProcessingLayer) .&&. register False controlDone)
