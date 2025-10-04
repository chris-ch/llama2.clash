module Model.Helpers.Seq.RowColCtrl
  ( RowColCtrlOut (..),
    rowColCtrl,
  )
where

import Clash.Prelude

-- Output bundle for one row drive and row scan
data RowColCtrlOut dom rows cols = RowColCtrlOut
  { clear :: Signal dom Bool, -- 1-cycle pulse at row start
    en :: Signal dom Bool, -- advance one column when True
    lastCol :: Signal dom Bool, -- asserted with en on final column
    colIdx :: Signal dom (Index cols), -- external copy of kernel's col
    rowIdx :: Signal dom (Index rows), -- current row (stable during row)
    rowDone :: Signal dom Bool, -- 1-cycle pulse (next cycle after lastCol)
    allDone :: Signal dom Bool -- 1-cycle pulse at final row completion
  }
  deriving (Generic, NFDataX)

-- FSM: Idle -> RowRun -> RowGap (commit) -> Idle (after last) or next RowRun
data S = Idle | RowRun | RowGap deriving (Generic, NFDataX, Eq)

-- Drives one row at a time. Start is a 1-cycle pulse.
-- Protocol:
--  * clear high for 1 cycle at row start.
--  * en high for exactly 'cols' cycles (lastCol asserted on final en).
--  * rowDone pulses 1 cycle after lastCol; rowIdx is still the completed row on that cycle.
--  * When the final row completes, allDone pulses for 1 cycle and controller idles.
rowColCtrl ::
  forall dom rows cols.
  (HiddenClockResetEnable dom, KnownNat rows, KnownNat cols) =>
  -- | start pulse
  Signal dom Bool ->
  RowColCtrlOut dom rows cols
rowColCtrl start = RowColCtrlOut {..}
  where
    -- Row index
    rowReg :: Signal dom (Index rows)
    rowReg = mealy goRow 0 (bundle (start, state, rowDone))
        where
        goRow r (st, s, rd) =
            let r' = case s of
                    Idle   -> if st then 0 else r
                    RowRun -> r
                    RowGap -> if rd then (if r == maxBound then r else succ r) else r
            in (r', r')
    rowIdx = rowReg

    -- State register
    state :: Signal dom S
    state = mealy goS Idle (bundle (start, rowDone, rowIdx))
      where
        goS s (st, rd, r) =
          let s' = case s of
                Idle -> if st then RowRun else Idle
                RowRun -> if rd then RowGap else RowRun
                RowGap -> if r == maxBound then Idle else RowRun
           in (s', s')

    -- Column counter mirrors kernel
    colReg :: Signal dom (Index cols)
    colReg = mealy goCol 0 (bundle (clear, en))
      where
        goCol c (cl, step) =
          let c0 = if cl then 0 else c
              cN = if step then (if c0 == maxBound then c0 else succ c0) else c0
           in (cN, c0)
    colIdx = colReg

    -- Control generation
    isRowRun = (== RowRun) <$> state
    en = isRowRun
    clear = regEn False isRowRun (pure True) -- 1 cycle high when RowRun begins
    lastCol = en .&&. ((== maxBound) <$> colIdx)
    rowDone = regEn False en lastCol
    allDone = rowDone .&&. ((== maxBound) <$> rowIdx)
