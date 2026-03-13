module LLaMa2.Layer.Attention.WeightLoader
  ( qWeightLoader      -- Q weights
  , kWeightLoader      -- K weights
  , vWeightLoader      -- V weights
  , woWeightLoader     -- WO output-projection weights
  , w1WeightLoader     -- FFN W1 (gate)
  , w2WeightLoader     -- FFN W2 (down)
  , w3WeightLoader     -- FFN W3 (up)
  , embWeightLoader    -- Vocabulary embedding (output projection)
  , WeightLoaderOutput(..)
  , LoadState(..)
  , assertRowStable
  ) where

import Clash.Prelude

import LLaMa2.Types.ModelConfig
    ( HeadDimension, ModelDimension, HiddenDimension, NumQueryHeads, NumLayers, NumKeyValueHeads, VocabularySize )
import LLaMa2.Numeric.Quantization (RowI8E (..))
import qualified LLaMa2.Memory.AXI.Slave as Slave
import qualified LLaMa2.Memory.AXI.Master as Master
import qualified LLaMa2.Memory.WeightsLayout as Layout
import qualified Prelude as P
import Data.Type.Bool (If)
import Data.Type.Ord (OrdCond)
import qualified GHC.TypeNats as T

data LoadState = LIdle | LFetching | LDone
  deriving (Show, Eq, Generic, NFDataX)

-- | Output bundle from a weight loader.  'numCols' is the column dimension
-- of the weight matrix (ModelDimension for Q/K/V, HeadDimension for WO).
data WeightLoaderOutput dom numCols = WeightLoaderOutput
  { dramRowOut        :: Signal dom (RowI8E numCols)
  , dbgRequestedAddr  :: Signal dom (Unsigned 32)
  , dbgCapturedAddr   :: Signal dom (Unsigned 32)
  , dbgCapturedRowReq :: Signal dom Int             -- row index as Int
  , dbgLoadState      :: Signal dom LoadState
  }

-- | Passthrough: returns rowSig unchanged (no cross-check assertions).
assertRowStable :: Signal dom Bool
  -> Signal dom (RowI8E n)
  -> Signal dom (RowI8E n)
assertRowStable _validSig rowSig = rowSig

-- | Weight loader for Q matrices (numRows=HeadDimension, numCols=ModelDimension)
qWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)  -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumQueryHeads
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool       -- ^ rowReqValid (pulse)
  -> Signal dom Bool       -- ^ downstreamReady (level)
  -> Signal dom Bool       -- ^ dataConsumed (pulse)
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom ModelDimension
     , Signal dom Bool     -- ^ weightValid (level)
     , Signal dom Bool     -- ^ weightReady (level)
     )
qWeightLoader cycleCounter dram layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.QMatrix layerIdx (fromIntegral headIdx)
                      rowReq rowReqValid downstreamReady dataConsumed "[WFU] "

-- | Weight loader for K matrices (numRows=HeadDimension, numCols=ModelDimension)
kWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumKeyValueHeads
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom ModelDimension
     , Signal dom Bool
     , Signal dom Bool
     )
kWeightLoader cycleCounter dram layerIdx kvHeadIdx rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.KMatrix layerIdx (fromIntegral kvHeadIdx)
                      rowReq rowReqValid downstreamReady dataConsumed "[KWFU] "

-- | Weight loader for V matrices (numRows=HeadDimension, numCols=ModelDimension)
vWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumKeyValueHeads
  -> Signal dom (Index HeadDimension)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom ModelDimension
     , Signal dom Bool
     , Signal dom Bool
     )
vWeightLoader cycleCounter dram layerIdx kvHeadIdx rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.VMatrix layerIdx (fromIntegral kvHeadIdx)
                      rowReq rowReqValid downstreamReady dataConsumed "[VWFU] "

-- | Weight loader for WO output-projection matrices.
-- WO is transposed vs Q/K/V: numRows=ModelDimension (64 rows), numCols=HeadDimension (8 cols).
woWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Index NumQueryHeads            -- ^ Q-head index (WO has NumQueryHeads heads)
  -> Signal dom (Index ModelDimension)  -- ^ row request (0..ModelDimension-1)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom HeadDimension
     , Signal dom Bool
     , Signal dom Bool
     )
woWeightLoader cycleCounter dram layerIdx headIdx rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.WOMatrix layerIdx (fromIntegral headIdx)
                      rowReq rowReqValid downstreamReady dataConsumed "[WOWFU] "

-- | Weight loader for W1 (gate) FFN matrices.
-- W1: MatI8E HiddenDimension ModelDimension — HiddenDimension rows × ModelDimension cols.
w1WeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index HiddenDimension)  -- ^ row request (0..HiddenDimension-1)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom ModelDimension
     , Signal dom Bool
     , Signal dom Bool
     )
w1WeightLoader cycleCounter dram layerIdx rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.W1Matrix layerIdx 0
                      rowReq rowReqValid downstreamReady dataConsumed "[W1WFU] "

-- | Weight loader for W3 (up) FFN matrices.
-- W3: MatI8E HiddenDimension ModelDimension — HiddenDimension rows × ModelDimension cols.
w3WeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index HiddenDimension)  -- ^ row request (0..HiddenDimension-1)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom ModelDimension
     , Signal dom Bool
     , Signal dom Bool
     )
w3WeightLoader cycleCounter dram layerIdx rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.W3Matrix layerIdx 0
                      rowReq rowReqValid downstreamReady dataConsumed "[W3WFU] "

-- | Weight loader for W2 (down) FFN matrices.
-- W2: MatI8E ModelDimension HiddenDimension — ModelDimension rows × HiddenDimension cols.
w2WeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index NumLayers)
  -> Signal dom (Index ModelDimension)   -- ^ row request (0..ModelDimension-1)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom HiddenDimension
     , Signal dom Bool
     , Signal dom Bool
     )
w2WeightLoader cycleCounter dram layerIdx rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.W2Matrix layerIdx 0
                      rowReq rowReqValid downstreamReady dataConsumed "[W2WFU] "

-- | Weight loader for vocabulary embedding rows (output/logits projection).
-- Shape: MatI8E VocabularySize ModelDimension — VocabularySize rows × ModelDimension cols.
-- Address: EmbeddingMatrix (at DRAM offset 0, independent of layer/head).
embWeightLoader :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Unsigned 32)
  -> Slave.AxiSlaveIn dom
  -> Signal dom (Index VocabularySize)   -- ^ row request (0..VocabularySize-1)
  -> Signal dom Bool
  -> Signal dom Bool
  -> Signal dom Bool
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom ModelDimension
     , Signal dom Bool
     , Signal dom Bool
     )
embWeightLoader cycleCounter dram rowReq rowReqValid downstreamReady dataConsumed =
  weightLoaderGeneric cycleCounter dram Layout.EmbeddingMatrix (pure 0) 0
                      rowReq rowReqValid downstreamReady dataConsumed "[EMBWFU] "

-- | Generic weight loader for any matrix type and dimensions.
--
-- Type parameters:
--   numRows — number of rows in the weight matrix (= HeadDimension for Q/K/V, ModelDimension for WO)
--   numCols — number of columns                   (= ModelDimension for Q/K/V, HeadDimension for WO)
--
-- Existing Q/K/V callers resolve (numRows=HeadDimension, numCols=ModelDimension) via type inference.
weightLoaderGeneric :: forall dom numRows numCols.
  ( HiddenClockResetEnable dom
  , KnownNat numRows
  , KnownNat numCols
  , KnownNat (Layout.WordsPerRow numCols)
  , KnownNat (If (OrdCond (CmpNat numCols 63) 'True 'True 'False) 1 (1 + Div numCols 64) T.* 64)
  )
  => Signal dom (Unsigned 32)       -- ^ cycleCounter for tracing
  -> Slave.AxiSlaveIn dom
  -> Layout.MatrixType              -- ^ Which matrix (QMatrix, KMatrix, VMatrix, WOMatrix)
  -> Signal dom (Index NumLayers)
  -> Int                            -- ^ Head index as Int
  -> Signal dom (Index numRows)     -- ^ Row request
  -> Signal dom Bool                -- ^ rowReqValid (pulse)
  -> Signal dom Bool                -- ^ downstreamReady (level)
  -> Signal dom Bool                -- ^ dataConsumed (pulse)
  -> String                         -- ^ Tag string for tracing
  -> ( Master.AxiMasterOut dom
     , WeightLoaderOutput dom numCols
     , Signal dom Bool              -- ^ weightValid (level)
     , Signal dom Bool              -- ^ weightReady (level)
     )
weightLoaderGeneric cycleCounter dram matrixType layerIdx headIdxInt rowReq rowReqValid downstreamReady dataConsumed tagStr =
  (axiMaster, out, weightValid, weightReady)
 where
  -- Loader FSM
  loadState :: Signal dom LoadState
  loadState = register LIdle nextState

  weightReady :: Signal dom Bool
  weightReady = loadState .==. pure LIdle

  weightValid :: Signal dom Bool
  weightValid = loadState .==. pure LDone

  -- Rising edge when a new row becomes valid to the downstream
  prevValid = register False weightValid
  dvRise    = weightValid .&&. (not <$> prevValid)

  -- Live request and address (combinational) — layerIdx is now a signal
  liveRow  :: Signal dom (Index numRows)
  liveRow  = rowReq

  liveAddr :: Signal dom (Unsigned 32)
  liveAddr = (\li ri -> Layout.rowAddressCalculator matrixType li headIdxInt (fromIntegral ri))
               <$> layerIdx <*> liveRow

  -- The actual fetch start: requires BOTH loader idle AND fetcher ready
  actualFetchStart :: Signal dom Bool
  actualFetchStart = weightReady .&&. fetcherReady .&&. rowReqValid

  -- AXI multi-word fetcher
  (axiMaster, fetchedWords, fetchValid, fetcherReady, fetcherDebug) =
    Layout.axiMultiWordRowFetcher @_ @numCols dram actualFetchStart liveAddr

  -- Transaction record captured on THE SAME handshake as the fetcher's address
  txnReg :: Signal dom (Txn numRows)
  txnReg = regEn (Txn 0 0) actualFetchStart (Txn <$> liveRow <*> liveAddr)

  -- ARADDR ASSERTION (passthrough)
  dramRowWithArCheck :: Signal dom (RowI8E numCols)
  dramRowWithArCheck = assertArAddrMatchGeneric
      cycleCounter
      (Layout.dbgArAccepted fetcherDebug)
      (Layout.dbgLatchedAddr fetcherDebug)
      (tAddr <$> txnReg)
      (tRow <$> txnReg)
      tagStr
      dramRowCommitted

  -- Parse and stage the DRAM row
  parsedRow :: Signal dom (RowI8E numCols)
  parsedRow = Layout.multiWordRowParser <$> fetchedWords

  zeroRow :: RowI8E numCols
  zeroRow = RowI8E { rowMantissas = repeat 0, rowExponent = 0 }

  -- Assemble on fetch completion; commit on dvRise (interface contract)
  dramRowAssembled :: Signal dom (RowI8E numCols)
  dramRowAssembled = regEn zeroRow fetchValid parsedRow

  dramRowCommitted :: Signal dom (RowI8E numCols)
  dramRowCommitted = regEn zeroRow dvRise dramRowAssembled

  dramRowWithTrace :: Signal dom (RowI8E numCols)
  dramRowWithTrace = traceRowIndicesGeneric
      cycleCounter
      actualFetchStart
      dvRise
      liveRow
      (tRow <$> txnReg)
      tagStr
      dramRowWithArCheck

  -- Loader FSM next-state
  nextState =
    mux (loadState .==. pure LIdle .&&. actualFetchStart)
        (pure LFetching)
    $ mux (loadState .==. pure LFetching .&&. fetchValid)
        (pure LDone)
    $ mux (loadState .==. pure LDone .&&. downstreamReady .&&. dataConsumed)
        (pure LIdle)
        loadState

  -- Outputs
  out :: WeightLoaderOutput dom numCols
  out = WeightLoaderOutput
    { dramRowOut        = dramRowWithTrace
    , dbgRequestedAddr  = liveAddr
    , dbgCapturedAddr   = tAddr <$> txnReg
    , dbgCapturedRowReq = fromIntegral . tRow <$> txnReg
    , dbgLoadState      = loadState
    }

--------------------------------------------------------------------------------
-- Helper functions (generalized over numRows / numCols)
--------------------------------------------------------------------------------

traceRowIndicesGeneric :: forall dom numRows numCols.
     Signal dom (Unsigned 32)           -- cycleCounter
  -> Signal dom Bool                    -- actualFetchStart
  -> Signal dom Bool                    -- dvRise
  -> Signal dom (Index numRows)         -- liveRow
  -> Signal dom (Index numRows)         -- txnReg.tRow
  -> String                             -- tag string
  -> Signal dom (RowI8E numCols)        -- pass-through signal
  -> Signal dom (RowI8E numCols)
traceRowIndicesGeneric _cyc _start _rise _lRow _txnRow _tagStr' rowIn = rowIn

-- | Passthrough: returns rowIn unchanged (no P.error assertions).
assertArAddrMatchGeneric :: forall dom numRows numCols.
     Signal dom (Unsigned 32)
  -> Signal dom Bool
  -> Signal dom (Unsigned 32)
  -> Signal dom (Unsigned 32)
  -> Signal dom (Index numRows)
  -> String
  -> Signal dom (RowI8E numCols)
  -> Signal dom (RowI8E numCols)
assertArAddrMatchGeneric _cyc _arAccepted _fetcherAddr _loaderAddr _rowIdx _tagStr' rowIn =
  rowIn

data Txn numRows = Txn
  { tRow  :: Index numRows
  , tAddr :: Unsigned 32
  } deriving (Generic, NFDataX, Show, Eq)
