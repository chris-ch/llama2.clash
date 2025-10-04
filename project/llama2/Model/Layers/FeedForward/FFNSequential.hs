module Model.Layers.FeedForward.FFNSequential
  ( runFFNSeq,
  )
where

import Clash.Prelude
import Model.Config (HiddenDimension, ModelDimension)
import Model.Helpers.FixedPoint (rmsNormFwFix)
import Model.Helpers.MatVecI8E_Seq (matVecRowSeq)
import Model.Helpers.Seq.RowColCtrl (RowColCtrlOut (..), rowColCtrl)
import Model.Layers.Components.Quantized (FeedForwardNetworkComponentQ (..))
import Model.Layers.FeedForward.FeedForwardNetwork.Internal (sigmoidLinearUnitF)
import Model.Numeric.ParamPack (QArray2D (..), RowI8E)
import Model.Numeric.Types (FixedPoint)

-- Sequential FFN with a single engine:
-- 1) xHat = RMSNorm(x, fRMSFfnF)
-- 2) gate = SiLU(W1 * xHat)           (HiddenDimension rows)
-- 3) up   =       W3 * xHat           (HiddenDimension rows)
-- 4) y    = W2 * (gate ⊙ up)          (ModelDimension rows)
-- 'start' is a 1-cycle pulse; 'done' pulses after final W2 row is committed.
runFFNSeq ::
  forall dom.
  (HiddenClockResetEnable dom) =>
  FeedForwardNetworkComponentQ ->
  -- | x (pre-attention residual)
  Signal dom (Vec ModelDimension FixedPoint) ->
  -- | start pulse (Stage4 entry)
  Signal dom Bool ->
  -- | done pulse
  ( Signal dom (Vec ModelDimension FixedPoint),
    -- \^ y
    Signal dom Bool
  )
runFFNSeq ffn x start = (yReg, doneFFN)
  where
    -- 1) Normalize once
    xHat :: Signal dom (Vec ModelDimension FixedPoint)
    xHat = rmsNormFwFix <$> x <*> pure (fRMSFfnF ffn)

    -- 2)+3) One controller for both W1 and W3 (same shape Hidden×Model)
    rcW1W3 :: RowColCtrlOut dom HiddenDimension ModelDimension
    rcW1W3 = rowColCtrl start

    -- W1 row stream and dot
    rowW1 :: Signal dom (RowI8E ModelDimension)
    rowW1 = (!!) (unQ2D (fW1Q ffn)) <$> rowIdx rcW1W3

    xColW1 :: Signal dom FixedPoint
    xColW1 = (!!) <$> xHat <*> colIdx rcW1W3

    (yRowW1, rowDoneW1) =
      matVecRowSeq (clear rcW1W3) (en rcW1W3) (lastCol rcW1W3) rowW1 xColW1

    gateBuf :: Signal dom (Vec HiddenDimension FixedPoint)
    gateBuf = mealy goG (repeat 0) (bundle (rowDoneW1, rowIdx rcW1W3, yRowW1))
      where
        goG acc (rd, r, y) =
          let acc' = if rd then replace r (sigmoidLinearUnitF y) acc else acc
           in (acc', acc')

    -- W3 row stream and dot (shares the same controller)
    rowW3 :: Signal dom (RowI8E ModelDimension)
    rowW3 = (!!) (unQ2D (fW3Q ffn)) <$> rowIdx rcW1W3

    xColW3 :: Signal dom FixedPoint
    xColW3 = xColW1

    (yRowW3, rowDoneW3) =
      matVecRowSeq (clear rcW1W3) (en rcW1W3) (lastCol rcW1W3) rowW3 xColW3

    upBuf :: Signal dom (Vec HiddenDimension FixedPoint)
    upBuf = mealy goU (repeat 0) (bundle (rowDoneW3, rowIdx rcW1W3, yRowW3))
      where
        goU acc (rd, r, y) =
          let acc' = if rd then replace r y acc else acc
           in (acc', acc')

    allDoneW3 :: Signal dom Bool
    allDoneW3 = allDone rcW1W3

    -- 4) Start W2 when W3 finishes (different shape Model×Hidden)
    rcW2 :: RowColCtrlOut dom ModelDimension HiddenDimension
    rcW2 = rowColCtrl allDoneW3

    zBuf :: Signal dom (Vec HiddenDimension FixedPoint)
    zBuf = zipWith (*) <$> gateBuf <*> upBuf

    rowW2 :: Signal dom (RowI8E HiddenDimension)
    rowW2 = (!!) (unQ2D (fW2Q ffn)) <$> rowIdx rcW2

    zCol :: Signal dom FixedPoint
    zCol = (!!) <$> zBuf <*> colIdx rcW2

    (yRowW2, rowDoneW2) =
      matVecRowSeq (clear rcW2) (en rcW2) (lastCol rcW2) rowW2 zCol

    yReg :: Signal dom (Vec ModelDimension FixedPoint)
    yReg = mealy goY (repeat 0) (bundle (rowDoneW2, rowIdx rcW2, yRowW2))
      where
        goY acc (rd, r, y) =
          let acc' = if rd then replace r y acc else acc
           in (acc', acc')

    doneFFN :: Signal dom Bool
    doneFFN = allDone rcW2
