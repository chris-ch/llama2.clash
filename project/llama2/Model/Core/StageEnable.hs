module Model.Core.StageEnable (StageEn(..), stageEnables) where

import Clash.Prelude
import Model.Core.Types (ProcessingState(..), CycleStage(..))

data StageEn = StageEn
  { enStage1 :: Bool
  , enStage2 :: Bool
  , enStage3 :: Bool
  , enStage4 :: Bool
  } deriving (Generic, NFDataX, Show, Eq)

stageEnables :: ProcessingState -> StageEn
stageEnables ps = case processingStage ps of
  Stage1_ProjectQKV  -> StageEn True  False False False
  Stage2_WriteKV     -> StageEn False True  False False
  Stage3_Attend      -> StageEn False False True  False
  Stage4_FeedForward -> StageEn False False False True
  _                  -> StageEn False False False False
