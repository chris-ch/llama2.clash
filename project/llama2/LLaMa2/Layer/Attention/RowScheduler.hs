module LLaMa2.Layer.Attention.RowScheduler
  ( RowSchedulerIn(..), RowSchedulerOut(..)
  , rowScheduler
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig (NumLayers, NumQueryHeads, HeadDimension)
import qualified Prelude as P
import Clash.Debug (trace)

-- | Trace row index changes
traceRowIndexChange :: Index NumLayers -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension) -> Signal dom (Index HeadDimension)
  -> Signal dom Bool -> Signal dom Bool -> Signal dom Bool
  -> Signal dom (Index HeadDimension)
traceRowIndexChange layerIdx headIdx current prev done ovl dsr = traced
  where
    traced = go <$> current <*> prev <*> done <*> ovl <*> dsr
    go curr p d o ds
      | curr /= p = trace (prefix P.++ "RI_CHANGE " P.++ show p P.++ "->" P.++ show curr 
                          P.++ " done=" P.++ show d P.++ " ovl=" P.++ show o P.++ " dsr=" P.++ show ds) curr
      | otherwise = curr
    prefix = "L" P.++ show layerIdx P.++ " H" P.++ show headIdx P.++ " "

data RowSchedulerIn dom = RowSchedulerIn
  { rsRowDone      :: Signal dom Bool  -- Row computation complete
  , rsOutputValid  :: Signal dom Bool  -- All rows complete
  , rsConsumeSignal :: Signal dom Bool  -- Coordinated consume from parent
  } deriving (Generic)

data RowSchedulerOut dom = RowSchedulerOut
  { rsRowIndex :: Signal dom (Index HeadDimension)  -- Current row being processed
  } deriving (Generic)

--------------------------------------------------------------------------------
-- COMPONENT: RowScheduler
-- Manages row index counter - increments on rowDone, resets on consume
--------------------------------------------------------------------------------
rowScheduler :: forall dom.
  HiddenClockResetEnable dom
  => Index NumLayers
  -> Index NumQueryHeads
  -> Signal dom Bool  -- downStreamReady for tracing
  -> RowSchedulerIn dom
  -> RowSchedulerOut dom
rowScheduler layerIdx headIdx downStreamReady inputs =
  RowSchedulerOut { rsRowIndex = rowIndexTraced }
  where
    rowIndex = register 0 nextRowIndex

    -- Increment on rowDone (unless at max), reset on consume
    nextRowIndex = 
      mux (rsRowDone inputs .&&. (rowIndex ./=. pure maxBound)) (rowIndex + 1)
      $ mux (rsOutputValid inputs .&&. rsConsumeSignal inputs) (pure 0)
        rowIndex

    rowIndexTraced = traceRowIndexChange layerIdx headIdx 
                       rowIndex (register 0 rowIndex)
                       (rsRowDone inputs) (rsOutputValid inputs) downStreamReady
