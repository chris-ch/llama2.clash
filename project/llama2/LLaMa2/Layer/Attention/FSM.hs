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
  Signal dom Bool ->  -- downStreamReady
  Signal dom Bool ->  -- computeDone
  ( Signal dom Bool   -- enable
  , Signal dom Bool   -- validOut
  , Signal dom Bool   -- readyForInput
  )
processingControllerFSM inValid downStreamReady computeDone = (enable, validOut, readyForInput)
 where
  state    = register PROCESSING_IDLE next
  readyForInput  = state .==. pure PROCESSING_IDLE
  startTx  = inValid .&&. readyForInput
  validOut = state .==. pure PROCESSING_DONE
  consume  = validOut .&&. downStreamReady

  next = mux (state .==. pure PROCESSING_IDLE)
              (mux startTx (pure PROCESSING_RUN) (pure PROCESSING_IDLE))
              (mux (state .==. pure PROCESSING_RUN)
                  (mux computeDone (pure PROCESSING_DONE) (pure PROCESSING_RUN))
                  (mux consume (pure PROCESSING_IDLE) (pure PROCESSING_DONE)))

  enable = startTx .||. (state .==. pure PROCESSING_RUN)

kvWriteControllerFSM ::
  (HiddenClockResetEnable dom) =>
  Signal dom Bool -> -- validIn (QKV done)
  Signal dom Bool -> -- downStreamReady (Attn ready)
  Signal dom Bool -> -- writeComplete (all banks written)
  ( Signal dom Bool, -- validOut (write done, ready for attn)
    Signal dom Bool, -- readyForInput (can accept QKV)
    Signal dom Bool -- enableWrite (trigger write)
  )
kvWriteControllerFSM validIn downStreamReady writeComplete = (validOut, readyForInput, enableWrite)
  where
    state = register PROCESSING_IDLE nextState

    readyForInput = state .==. pure PROCESSING_IDLE
    startWrite = validIn .&&. readyForInput
    validOut = state .==. pure PROCESSING_DONE
    consume = validOut .&&. downStreamReady

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
