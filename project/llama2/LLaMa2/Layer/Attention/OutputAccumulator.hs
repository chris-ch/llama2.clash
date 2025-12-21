module LLaMa2.Layer.Attention.OutputAccumulator
  ( outputAccumulator, OutputAccumIn(..), OutputAccumOut(..)
  ) where

import Clash.Prelude
import LLaMa2.Numeric.Types
import LLaMa2.Types.ModelConfig
import qualified Prelude as P
import Clash.Debug (trace)

-- | Trace accumulator updates
traceAccumUpdate :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom Bool -> Signal dom (Index HeadDimension) -> Signal dom FixedPoint
  -> Signal dom a -> Signal dom a
traceAccumUpdate layerIdx headIdx done ri value current = traced
  where
    traced = go <$> done <*> ri <*> value <*> current
    go d ridx val curr
      | d         = trace (prefix P.++ "QOUT_ACCUM ri=" P.++ show ridx P.++ " val=" P.++ show val) curr
      | otherwise = curr
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

--------------------------------------------------------------------------------
-- COMPONENT: OutputAccumulator
-- Accumulates row results into output vector
-- NO FEEDBACK LOOPS - safe to extract
--------------------------------------------------------------------------------
data OutputAccumIn dom = OutputAccumIn
  { oaRowDone     :: Signal dom Bool
  , oaRowIndex    :: Signal dom (Index HeadDimension)
  , oaRowResult   :: Signal dom FixedPoint
  , oaRowResultHC :: Signal dom FixedPoint
  } deriving (Generic)

data OutputAccumOut dom = OutputAccumOut
  { oaOutput   :: Signal dom (Vec HeadDimension FixedPoint)
  , oaOutputHC :: Signal dom (Vec HeadDimension FixedPoint)
  } deriving (Generic)

outputAccumulator :: forall dom.
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Index NumQueryHeads
  -> OutputAccumIn dom
  -> OutputAccumOut dom
outputAccumulator layerIdx headIdx inputs =
  OutputAccumOut
    { oaOutput   = qOut
    , oaOutputHC = qOutHC
    }
  where
    -- DRAM result accumulator
    qOut = register (repeat 0) nextOutput
    
    qOutTraced = traceAccumUpdate layerIdx headIdx 
                   (oaRowDone inputs) (oaRowIndex inputs) 
                   (oaRowResult inputs) qOut

    nextOutput = mux (oaRowDone inputs)
                     (replace <$> oaRowIndex inputs <*> oaRowResult inputs <*> qOutTraced)
                     qOut

    -- HC reference accumulator
    qOutHC = register (repeat 0) nextOutputHC

    nextOutputHC = mux (oaRowDone inputs)
                       (replace <$> oaRowIndex inputs <*> oaRowResultHC inputs <*> qOutHC)
                       qOutHC
