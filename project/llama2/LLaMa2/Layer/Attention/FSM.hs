module LLaMa2.Layer.Attention.FSM (
    processingControllerFSM, SingleHeadState(..), kvWriteControllerFSM
) where
import Clash.Prelude

-- FSM states for single head controller
data SingleHeadState = SINGLE_HEAD_IDLE | SINGLE_HEAD_REQUESTING | SINGLE_HEAD_PROJECTING | SINGLE_HEAD_DONE
  deriving (Eq, Show, Generic, NFDataX)

-- Shared 3-state valid/ready controller
data GenericState = PROCESSING_IDLE | PROCESSING_RUN | PROCESSING_DONE
  deriving (Show, Eq, Generic, NFDataX)

processingControllerFSM ::
  HiddenClockResetEnable dom =>
  Signal dom Bool ->  -- inValid
  Signal dom Bool ->  -- outReady
  Signal dom Bool ->  -- computeDone
  ( Signal dom Bool   -- enable
  , Signal dom Bool   -- validOut
  , Signal dom Bool   -- inReady
  )
processingControllerFSM inValid outReady computeDone = (enable, validOut, inReady)
 where
  state    = register PROCESSING_IDLE next
  inReady  = state .==. pure PROCESSING_IDLE
  startTx  = inValid .&&. inReady
  validOut = state .==. pure PROCESSING_DONE
  consume  = validOut .&&. outReady

  next = mux (state .==. pure PROCESSING_IDLE)
              (mux startTx (pure PROCESSING_RUN) (pure PROCESSING_IDLE))
              (mux (state .==. pure PROCESSING_RUN)
                  (mux computeDone (pure PROCESSING_DONE) (pure PROCESSING_RUN))
                  (mux consume (pure PROCESSING_IDLE) (pure PROCESSING_DONE)))

  enable = startTx .||. (state .==. pure PROCESSING_RUN)

kvWriteControllerFSM ::
  (HiddenClockResetEnable dom) =>
  Signal dom Bool -> -- validIn (QKV done)
  Signal dom Bool -> -- readyOut (Attn ready)
  Signal dom Bool -> -- writeComplete (all banks written)
  ( Signal dom Bool, -- validOut (write done, ready for attn)
    Signal dom Bool, -- readyIn (can accept QKV)
    Signal dom Bool -- enableWrite (trigger write)
  )
kvWriteControllerFSM validIn readyOut writeComplete = (validOut, readyIn, enableWrite)
  where
    state = register PROCESSING_IDLE nextState

    readyIn = state .==. pure PROCESSING_IDLE
    startWrite = validIn .&&. readyIn
    validOut = state .==. pure PROCESSING_DONE
    consume = validOut .&&. readyOut

    nextState =
      mux
        (state .==. pure PROCESSING_IDLE)
        (mux startWrite (pure PROCESSING_RUN) (pure PROCESSING_IDLE))
        ( mux
            (state .==. pure PROCESSING_RUN)
            (mux writeComplete (pure PROCESSING_DONE) (pure PROCESSING_RUN))
            (mux consume (pure PROCESSING_IDLE) (pure PROCESSING_DONE))
        )

    enableWrite = startWrite .||. (state .==. pure PROCESSING_RUN)
