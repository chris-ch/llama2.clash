{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use fewer imports" #-}
module LLaMa2.Layers.Attention.MultiHeadAttention (
  projectQKV
) where

import Clash.Prelude
import LLaMa2.Config
  ( NumQueryHeads, ModelDimension, NumKeyValueHeads
  , HeadDimension, SequenceLength)

import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Layers.Components.Quantized
  ( MultiHeadAttentionComponentQ(..) )
import LLaMa2.Layers.Attention.MultiHeadAttention.Internal
  ( computeHeadQ, computeHeadKV )
import LLaMa2.Helpers.FixedPoint (rmsNormFwFix)
import LLaMa2.Helpers (liftA5)


-- State machine for Option B: Q group then KV group
data QKVSeqState 
  = QKVIdle
  | QKVComputingQ     -- All Q heads in parallel
  | QKVComputingKV    -- All KV heads in parallel
  | QKVDone
  deriving (Show, Eq, Generic, NFDataX)

-- Sequential QKV projection with handshaking
projectQKV
  :: forall dom
   . HiddenClockResetEnable dom
  => Signal dom Bool                              -- validIn
  -> Signal dom Bool                              -- readyIn (downstream)
  -> MultiHeadAttentionComponentQ
  -> Signal dom (Index SequenceLength)
  -> Signal dom (Vec ModelDimension FixedPoint)
  -> ( Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
     , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
     , Signal dom Bool                            -- validOut
     , Signal dom Bool                            -- readyOut
     )
projectQKV validIn readyInDownstream mha stepCount inputVector =
  (queriesOut, keysOut, valuesOut, validOut, readyOut)
  where
    -- Normalize input
    normalizedInput = rmsNormFwFix <$> inputVector <*> pure (rmsAttF mha)
    
    -- ===== State Machine =====
    state :: Signal dom QKVSeqState
    state = register QKVIdle nextState
    
    -- Rising edge detection for validIn
    validInPrev = register False validIn
    startPulse = validIn .&&. (not <$> validInPrev)
    
    nextState :: Signal dom QKVSeqState
    nextState = liftA5 stateTransition state startPulse allQReady allKVReady readyInDownstream
    
    stateTransition :: QKVSeqState -> Bool -> Bool -> Bool -> Bool -> QKVSeqState
    stateTransition QKVIdle start _ _ _ = 
      if start then QKVComputingQ else QKVIdle
    stateTransition QKVComputingQ _ allQDone _ _ = 
      if allQDone then QKVComputingKV else QKVComputingQ
    stateTransition QKVComputingKV _ _ allKVDone _ = 
      if allKVDone then QKVDone else QKVComputingKV
    stateTransition QKVDone _ _ _ downstreamReady = 
      -- downstreamReady is HIGH when downstream has consumed (stage has ended)
      if downstreamReady then QKVIdle else QKVDone
    
    -- ===== Compute All Q Heads in Parallel =====
    validInQ = liftA2 (==) state (pure QKVComputingQ)
    readyDownQ = pure True  -- Internal signal, always ready
    
    qResults :: Vec NumQueryHeads 
                   ( Signal dom (Vec HeadDimension FixedPoint)
                   , Signal dom Bool
                   , Signal dom Bool )
    qResults = imap (\qIx _ ->
      let headQ = headsQ mha !! qIx
      in computeHeadQ validInQ readyDownQ headQ stepCount normalizedInput
      ) indicesI
    
    -- Extract Q results and ready signals
    qVectors :: Vec NumQueryHeads (Signal dom (Vec HeadDimension FixedPoint))
    qVectors = map (\(vec, _, _) -> vec) qResults
    
    qReadySignals :: Vec NumQueryHeads (Signal dom Bool)
    qReadySignals = map (\(_, _, ready) -> ready) qResults
    
    -- All Q heads ready when all individual ready signals are high
    allQReady :: Signal dom Bool
    allQReady = foldl (.&&.) (pure True) qReadySignals
    
    -- ===== Compute All KV Heads in Parallel =====
    validInKV = liftA2 (==) state (pure QKVComputingKV)
    readyDownKV = pure True  -- Internal signal, always ready
    
    kvResults :: Vec NumKeyValueHeads 
                    ( Signal dom (Vec HeadDimension FixedPoint)
                    , Signal dom (Vec HeadDimension FixedPoint)
                    , Signal dom Bool
                    , Signal dom Bool )
    kvResults = imap (\kvIx _ ->
      let nQ = natToNum @NumQueryHeads :: Int
          nKV = natToNum @NumKeyValueHeads :: Int
          qIdx0 = toEnum (min (nQ - 1) (fromEnum kvIx * (nQ `div` nKV))) :: Index NumQueryHeads
          headQ = headsQ mha !! qIdx0
      in computeHeadKV validInKV readyDownKV headQ stepCount normalizedInput
      ) indicesI
    
    -- Extract KV results and ready signals
    kVectors :: Vec NumKeyValueHeads (Signal dom (Vec HeadDimension FixedPoint))
    kVectors = map (\(k, _, _, _) -> k) kvResults
    
    vVectors :: Vec NumKeyValueHeads (Signal dom (Vec HeadDimension FixedPoint))
    vVectors = map (\(_, v, _, _) -> v) kvResults
    
    kvReadySignals :: Vec NumKeyValueHeads (Signal dom Bool)
    kvReadySignals = map (\(_, _, _, ready) -> ready) kvResults
    
    -- All KV heads ready when all individual ready signals are high
    allKVReady :: Signal dom Bool
    allKVReady = foldl (.&&.) (pure True) kvReadySignals
    
    -- ===== Hold Results Until Done =====
    -- Latch Q results when Q computation completes
    qResultsLatched :: Signal dom (Vec NumQueryHeads (Vec HeadDimension FixedPoint))
    qResultsLatched = regEn 
      (repeat (repeat 0))
      allQReady
      (sequenceA qVectors)
    
    -- Latch KV results when KV computation completes
    kResultsLatched :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
    kResultsLatched = regEn 
      (repeat (repeat 0))
      allKVReady
      (sequenceA kVectors)
    
    vResultsLatched :: Signal dom (Vec NumKeyValueHeads (Vec HeadDimension FixedPoint))
    vResultsLatched = regEn 
      (repeat (repeat 0))
      allKVReady
      (sequenceA vVectors)
    
    -- ===== Outputs =====
    isDone = liftA2 (==) state (pure QKVDone)
    
    queriesOut = mux isDone qResultsLatched (pure (repeat (repeat 0)))
    keysOut = mux isDone kResultsLatched (pure (repeat (repeat 0)))
    valuesOut = mux isDone vResultsLatched (pure (repeat (repeat 0)))
    
    -- Valid pulse when entering Done state
    statePrev = register QKVIdle state
    validOut = liftA2 (\curr prev -> curr == QKVDone && prev == QKVComputingKV)
                      state statePrev
    
    -- Ready signal (stable high in Done state)
    readyOut = isDone
