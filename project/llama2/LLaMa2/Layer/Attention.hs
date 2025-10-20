module LLaMa2.Layer.Attention (
    fsmController
) where
import Clash.Prelude

-- Shared 3-state valid/ready controller
data GenericState = Idle | Compute | Done
  deriving (Show, Eq, Generic, NFDataX)

fsmController ::
  HiddenClockResetEnable dom =>
  Signal dom Bool ->  -- inValid
  Signal dom Bool ->  -- outReady
  Signal dom Bool ->  -- computeDone
  ( Signal dom Bool   -- enable
  , Signal dom Bool   -- validOut
  , Signal dom Bool   -- inReady
  )
fsmController inValid outReady computeDone = (enable, validOut, inReady)
 where
  state    = register Idle next
  inReady  = state .==. pure Idle
  startTx  = inValid .&&. inReady
  validOut = state .==. pure Done
  consume  = validOut .&&. outReady

  next = mux (state .==. pure Idle)
              (mux startTx (pure Compute) (pure Idle))
              (mux (state .==. pure Compute)
                  (mux computeDone (pure Done) (pure Compute))
                  (mux consume (pure Idle) (pure Done)))

  enable = startTx .||. (state .==. pure Compute)
