module LLaMa2.Embedding.InputEmbedding
 ( inputEmbedding
) where
import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( ModelDimension )
import LLaMa2.Types.LayerData ( Token )
import LLaMa2.Numeric.Types ( FixedPoint, scalePow2F )
import LLaMa2.Numeric.Quantization ( RowI8E (..) )
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout

--------------------------------------------------------------------------------
-- State machine
--------------------------------------------------------------------------------
data EmbedState = EmbIdle | EmbFetching | EmbReady
  deriving (Generic, NFDataX, Show, Eq)

--------------------------------------------------------------------------------
-- | Dequantize a RowI8E into a Vec of FixedPoint.
--------------------------------------------------------------------------------
deqRow :: RowI8E n -> Vec n FixedPoint
deqRow RowI8E { rowMantissas = mant, rowExponent = e } =
  let s = scalePow2F e 1
  in map (\q -> fromIntegral q * s) mant

--------------------------------------------------------------------------------
-- | DRAM-backed token embedding lookup.
--
-- On each fetchTrigger pulse (outputToken must be stable):
--   1. Issues an AXI multi-word burst to fetch the vocabulary row for tokenIdx.
--   2. Parses and dequantizes the result.
--   3. Holds the result until the next fetch.
--
-- outputValid: False until the first fetch completes; True thereafter.
-- isBusy:      True while an AXI fetch is in flight.
--------------------------------------------------------------------------------
{-# NOINLINE inputEmbedding #-}
inputEmbedding :: forall dom.
  HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)                       -- ^ cycleCounter
  -> Slave.AxiSlaveIn dom                           -- ^ DRAM
  -> Signal dom Bool                                -- ^ fetchTrigger (1-cycle pulse)
  -> Signal dom Token                               -- ^ token index (stable when trigger fires)
  -> ( Master.AxiMasterOut dom
     , Signal dom (Vec ModelDimension FixedPoint)   -- ^ embedded output (holds last result)
     , Signal dom Bool                              -- ^ outputValid
     , Signal dom Bool                              -- ^ isBusy
     )
inputEmbedding _cycleCounter dramSlaveIn fetchTrigger tokenSig =
  (axiMaster, outputVec, outputValid, isBusy)
 where
  -- -----------------------------------------------------------------------
  -- DRAM multi-word fetcher (WordsPerRow ModelDimension beats per request)
  -- -----------------------------------------------------------------------
  fetchAddr :: Signal dom (Unsigned 32)
  fetchAddr = Layout.embeddingRowAddress . fromIntegral <$> tokenSig

  -- Buffer the trigger through a skid stage so requests survive the
  -- reset cycle when axiMultiWordRowFetcher is not yet ready.
  (captureAvail, capturedFetchAddr) =
    Layout.requestCaptureStage fetchTrigger fetchAddr fetcherReady

  (axiMaster, wordsOut, fetchDone, fetcherReady, _dbg) =
    Layout.axiMultiWordRowFetcher @dom @ModelDimension dramSlaveIn captureAvail capturedFetchAddr

  -- -----------------------------------------------------------------------
  -- State machine
  -- -----------------------------------------------------------------------
  state :: Signal dom EmbedState
  state = register EmbIdle nextState

  -- A new fetch starts the cycle captureAvail && fetcherReady fires
  newFetchStarting :: Signal dom Bool
  newFetchStarting = captureAvail .&&. fetcherReady .&&. (state ./=. pure EmbFetching)

  nextState :: Signal dom EmbedState
  nextState =
    mux newFetchStarting (pure EmbFetching) $
    mux (state .==. pure EmbFetching .&&. fetchDone) (pure EmbReady) state

  -- -----------------------------------------------------------------------
  -- Parse and dequantize on fetch completion
  -- -----------------------------------------------------------------------
  dramRow :: Signal dom (RowI8E ModelDimension)
  dramRow = Layout.multiWordRowParser <$> wordsOut

  dramVec :: Signal dom (Vec ModelDimension FixedPoint)
  dramVec = deqRow <$> dramRow

  -- -----------------------------------------------------------------------
  -- Output register: latch on rising edge of fetchDone (EmbFetching->EmbReady)
  -- -----------------------------------------------------------------------
  capturing :: Signal dom Bool
  capturing = state .==. pure EmbFetching .&&. fetchDone

  outputVec :: Signal dom (Vec ModelDimension FixedPoint)
  outputVec = register (repeat 0) $ mux capturing dramVec outputVec

  outputValid :: Signal dom Bool
  outputValid = state .==. pure EmbReady .&&. (not <$> newFetchStarting)

  isBusy :: Signal dom Bool
  isBusy = state .==. pure EmbFetching .||. newFetchStarting
