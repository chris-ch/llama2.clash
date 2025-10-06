module Model.Layers.TransformerLayer.Internal (
    controlOneHead
) where

import Clash.Prelude
import Model.Config (ModelDimension , HeadDimension)


import Model.Helpers.MatVecI8E (sequentialMatVecStub)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.ParamPack (QArray2D)

-- Controller for one head's WO projection
controlOneHead ::
  forall dom .
  HiddenClockResetEnable dom
  => Signal dom (Vec HeadDimension FixedPoint)           -- head output
  -> Signal dom Bool                                      -- head done
  -> QArray2D ModelDimension HeadDimension               -- WO matrix
  -> ( Signal dom (Vec ModelDimension FixedPoint)        -- projected output
     , Signal dom Bool                                    -- validOut
     , Signal dom Bool                                    -- readyOut
     )
controlOneHead headOutput headDone woMatrix = (projOut, validOut, readyOut)
  where
    -- Detect rising edge of headDone
    headDonePrev = register False headDone
    headDoneRising = headDone .&&. (not <$> headDonePrev)
    
    -- State: IDLE (0) -> PROJECTING (1) -> DONE (2)
    state :: Signal dom (Unsigned 2)
    state = register 0 nextState
    
    nextState = 
      mux (state .==. 0 .&&. headDoneRising) (pure 1) $  -- Rising edge of headDone -> start
      mux (state .==. 1 .&&. woValidOut) (pure 2) $      -- WO done -> done state
      mux (state .==. 2) (pure 0)                        -- Reset to idle next cycle
      state                                              -- Hold state
    
    -- Start WO projection when entering state 1
    startWO = state .==. 0 .&&. headDoneRising
    
    -- Call the sequential matmul stub (no latching - use current headOutput)
    (woResult, woValidOut, woReadyOut) = 
      sequentialMatVecStub woMatrix (bundle (startWO, headOutput))
    
    -- Output the result (hold it when valid)
    projOut = regEn (repeat 0) woValidOut woResult
    
    -- Valid out in DONE state (state 2)
    validOut = state .==. 2
    
    -- Ready when idle
    readyOut = state .==. 0
