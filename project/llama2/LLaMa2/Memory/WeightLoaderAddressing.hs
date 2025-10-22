module LLaMa2.Memory.WeightLoaderAddressing
  ( WeightMatrixType(..)
  , WeightAddress(..)
  , weightAddressGenerator
  ) where

import Clash.Prelude
import LLaMa2.Types.ModelConfig
  ( HeadDimension
  , NumQueryHeads
  , NumKeyValueHeads
  )

-- Q/K/V discriminator for streamed rows
data WeightMatrixType = QMatrix | KMatrix | VMatrix
  deriving (Generic, NFDataX, Show, Eq, Enum, Bounded)

-- Self-describing row address
-- rowIndex: 0..(HeadDimension-1)
-- matrixType: Q, K, or V
-- headIndex: head number (fits in 8 bits for our supported configs)
data WeightAddress = WeightAddress
  { rowIndex   :: Index HeadDimension
  , matrixType :: WeightMatrixType
  , headIndex  :: Unsigned 8
  } deriving (Generic, NFDataX, Show, Eq)

-- Helpers: compile-time constants for number of heads (as Unsigned 8)
qHeadCountU8 :: Unsigned 8
qHeadCountU8 = fromInteger (natToNum @NumQueryHeads)

kvHeadCountU8 :: Unsigned 8
kvHeadCountU8 = fromInteger (natToNum @NumKeyValueHeads)

-- How many heads are in the current matrix family?
headLimitFor :: WeightMatrixType -> Unsigned 8
headLimitFor QMatrix = qHeadCountU8
headLimitFor KMatrix = kvHeadCountU8
headLimitFor VMatrix = kvHeadCountU8

-- True for one cycle at the end of (Q, all heads, all rows)
-- then (K, ...), then (V, ...). We count rows only when enable is True.
-- Reset restarts the sequence at (row=0, Q, head=0).
weightAddressGenerator :: forall dom .
  HiddenClockResetEnable dom
  => Signal dom Bool              -- ^ enable (streamValid)
  -> Signal dom Bool              -- ^ reset (layer change / new load)
  -> ( Signal dom WeightAddress
     , Signal dom Bool )          -- ^ allDone (pulse when V last head last row completes)
weightAddressGenerator enable reset = (addrOut, allDonePulse)
  where
    -- State registers
    rowReg   :: Signal dom (Index HeadDimension)
    rowReg   = register 0 rowNext

    matReg   :: Signal dom WeightMatrixType
    matReg   = register QMatrix matNext

    headReg  :: Signal dom (Unsigned 8)
    headReg  = register 0 headNext

    -- Current terminal conditions
    rowIsLast  = rowReg .==. pure maxBound
    headIsLast = headReg .==. (pred <$> fmap headLimitFor matReg)
    matIsV     = matReg .==. pure VMatrix

    -- A row completes when we are enabled and at last row (consumed this cycle)
    rowStep   = enable
    rowWrap   = rowStep .&&. rowIsLast

    -- Done for entire QKV set: last row of last head in V matrix
    doneNow   = rowStep .&&. rowIsLast .&&. headIsLast .&&. matIsV
    allDonePulse = register False (mux reset (pure False) doneNow)

    -- Next row
    rowNext = mux reset
                (pure 0)
                (mux rowStep
                  (mux rowIsLast (pure 0) (rowReg + 1))
                  rowReg)

    -- Head increments only when a row wraps
    headNext = mux reset
                 (pure 0)
                 (mux rowWrap
                   (mux headIsLast (pure 0) (headReg + 1))
                   headReg)

    -- Matrix changes only when a row wraps and the current head last wraps too
    matNext = mux reset
                (pure QMatrix)
                (mux (rowWrap .&&. headIsLast)
                  (caseMat <$> matReg)
                  matReg)

    caseMat QMatrix = KMatrix
    caseMat KMatrix = VMatrix
    caseMat VMatrix = QMatrix  -- auto-wrap to Q for next layer

    addrOut = WeightAddress <$> rowReg <*> matReg <*> headReg
