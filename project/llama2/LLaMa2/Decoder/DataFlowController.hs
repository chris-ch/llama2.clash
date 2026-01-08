module LLaMa2.Decoder.DataFlowController
  ( DataFlowController(..)
  , dataFlowController
  , DataStage(..)
  ) where

import Clash.Prelude
import qualified Prelude as P
import LLaMa2.Types.ModelConfig (NumLayers, SequenceLength)
import Clash.Debug (trace)

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
  => Signal dom (Unsigned 32)  -- ^ cycleCounter for tracing
  -> Signal dom Bool           -- ^ ffnDone: layer processing complete
  -> Signal dom Bool           -- ^ classifierDone: token classification complete
  -> DataFlowController dom
dataFlowController cycleCounter ffnDone classifierDone =
  DataFlowController
    { processingStage = stageSTraced
    , currentLayer    = layerSTraced
    , seqPosition     = seqPosSTraced
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

    -- Trace seqPos changes
    seqPosPrev :: Signal dom (Index SequenceLength)
    seqPosPrev = register 0 seqPosS

    seqPosChanged :: Signal dom Bool
    seqPosChanged = seqPosS ./=. seqPosPrev

    seqPosSTraced :: Signal dom (Index SequenceLength)
    seqPosSTraced = traceSeqPos <$> cycleCounter <*> seqPosChanged <*> seqPosPrev <*> seqPosS
      where
        traceSeqPos cyc changed oldPos newPos
          | changed   = trace ("@" P.++ show cyc P.++ " [DFC] seqPos: " 
                               P.++ show oldPos P.++ " -> " P.++ show newPos) newPos
          | otherwise = newPos

    -- Trace layer changes
    layerPrev :: Signal dom (Index NumLayers)
    layerPrev = register 0 layerS

    layerChanged :: Signal dom Bool
    layerChanged = layerS ./=. layerPrev

    layerSTraced :: Signal dom (Index NumLayers)
    layerSTraced = traceLayer <$> cycleCounter <*> layerChanged <*> layerPrev <*> layerS
      where
        traceLayer cyc changed oldLayer newLayer
          | changed   = trace ("@" P.++ show cyc P.++ " [DFC] layer: " 
                               P.++ show oldLayer P.++ " -> " P.++ show newLayer) newLayer
          | otherwise = newLayer

    -- Trace stage changes
    stagePrev :: Signal dom DataStage
    stagePrev = register ProcessingLayer stageS

    stageChanged :: Signal dom Bool
    stageChanged = stageS ./=. stagePrev

    stageSTraced :: Signal dom DataStage
    stageSTraced = traceStage <$> cycleCounter <*> stageChanged <*> stagePrev <*> stageS
      where
        traceStage cyc changed oldStage newStage
          | changed   = trace ("@" P.++ show cyc P.++ " [DFC] stage: " 
                               P.++ show oldStage P.++ " -> " P.++ show newStage) newStage
          | otherwise = newStage

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
