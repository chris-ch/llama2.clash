module Model.Core.LayerController
  ( LayerIO(..)
  , LayerCtrl(..)
  , layerStep
  ) where

import Clash.Prelude
import Model.Core.Types (ProcessingState(..), CycleStage(..), SequenceLength)

data LayerIO = LayerIO
  { writeDone :: Bool   -- Stage2 done
  , attnDone  :: Bool   -- Stage3 done
  , prevFFNOK :: Bool   -- upstream layer finished Stage4 for this pos
  , haveToken :: Bool   -- layer 0: token available (prefill or sampled)
  } deriving (Generic, NFDataX)

data LayerCtrl = LayerCtrl
  { ps      :: ProcessingState
  , fireS1  :: Bool
  , fireS3  :: Bool
  , doneF4  :: Bool
  } deriving (Generic, NFDataX)

layerStep :: ProcessingState -> LayerIO -> (ProcessingState, LayerCtrl)
layerStep s io =
  let s1 = case processingStage s of
        Stage1_ProjectQKV  -> s { processingStage = Stage2_WriteKV }
        Stage2_WriteKV     -> if writeDone io then s { processingStage = Stage3_Attend } else s
        Stage3_Attend      -> if attnDone  io then s { processingStage = Stage4_FeedForward } else s
        Stage4_FeedForward -> s { processingStage = Stage5_Bookkeeping }
        Stage5_Bookkeeping -> s { processingStage  = Stage1_ProjectQKV
                                 , sequencePosition = if sequencePosition s == maxBound then 0 else succ (sequencePosition s) }
  in ( s1
     , LayerCtrl { ps     = s1
                 , fireS1 = processingStage s  /= Stage1_ProjectQKV
                         && processingStage s1 == Stage1_ProjectQKV
                         && (haveToken io || prevFFNOK io)
                 , fireS3 = processingStage s  /= Stage3_Attend
                         && processingStage s1 == Stage3_Attend
                 , doneF4 = processingStage s  == Stage4_FeedForward
                         && processingStage s1 == Stage5_Bookkeeping
                 })
