module LLaMa2.Layers.FeedForward.FeedForwardNetwork (
   feedForwardStageSeq
) where

import Clash.Prelude
import LLaMa2.Helpers.FixedPoint ( rmsNormFwFix )
import LLaMa2.Config
    ( ModelDimension, ModelDimension )
import LLaMa2.Numeric.Types ( FixedPoint, FixedPoint )
import LLaMa2.Layers.Components.Quantized
    ( FeedForwardNetworkComponentQ(fRMSFfnF) )
import LLaMa2.Layers.FeedForward.FeedForwardNetwork.Internal (feedForwardCoreSeq)

feedForwardStageSeq
  :: HiddenClockResetEnable dom
  => Signal dom Bool                              -- ^ validIn
  -> Signal dom Bool                              -- ^ readyIn (from downstream)
  -> FeedForwardNetworkComponentQ
  -> Signal dom (Vec ModelDimension FixedPoint)   -- ^ input vector
  -> ( Signal dom (Vec ModelDimension FixedPoint) -- ^ output vector
     , Signal dom Bool                             -- ^ validOut
     , Signal dom Bool                             -- ^ readyOut (to upstream)
     )
feedForwardStageSeq validIn readyIn ffn inputVector =
  (outputVector, validOut, readyOut)
  where
    -- Pre-normalize the input (combinational)
    xHat = rmsNormFwFix <$> inputVector <*> pure (fRMSFfnF ffn)
    
    -- Sequential FFN core with handshaking
    (ffnCore, coreValidOut, coreReadyOut) =
      feedForwardCoreSeq validIn readyIn ffn xHat
    
    -- Add residual connection when core output is valid
    -- Register the residual to align with FFN output timing
    inputVectorDelayed = regEn (repeat 0) coreValidOut inputVector
    outputVector = zipWith (+) <$> inputVectorDelayed <*> ffnCore
    
    -- Pass through handshaking signals
    validOut = coreValidOut
    readyOut = coreReadyOut
