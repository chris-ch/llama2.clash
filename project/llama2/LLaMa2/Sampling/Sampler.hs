module LLaMa2.Sampling.Sampler
  ( tokenSampler, samplerTop
  ) where

import Clash.Prelude
import LLaMa2.Types.LayerData (Temperature, Token, Seed)
import LLaMa2.Types.ModelConfig (VocabularySize)
import LLaMa2.Numeric.Types (FixedPoint)
import LLaMa2.Numeric.Quantization (expF)
import LLaMa2.Sampling.Distribution (uniformRandom01Generator)

-- ---------------------------------------------------------------------------
-- Internal FSM
-- ---------------------------------------------------------------------------
data SamplerState
  = SIdle
  | SFill      -- receive streaming logits, write BRAM, track argmax
  | SExpScan   -- pass 2: read logit[i], write exp[i], accumulate sum
  | SCDFScan   -- pass 3: read exp[i], walk CDF, sample token
  | SDone
  deriving (Generic, NFDataX, Show, Eq)

-- ---------------------------------------------------------------------------
-- | Sequential BRAM-backed token sampler.
--
-- Receives logits one per cycle from logitsProjector (streaming interface).
-- Supports greedy argmax (temperature == 0) and softmax categorical sampling
-- (temperature > 0).
--
-- Latency after logitsAllDone:
--   Argmax:   1 cycle
--   Sampling: ~2 × VocabularySize cycles (two BRAM passes)
-- ---------------------------------------------------------------------------
{-# NOINLINE tokenSampler #-}
tokenSampler :: forall dom. HiddenClockResetEnable dom
  => Signal dom (Index VocabularySize)  -- ^ logit index   (from logitsProjector)
  -> Signal dom FixedPoint              -- ^ logit value   (from logitsProjector)
  -> Signal dom Bool                    -- ^ logitValid    (one pulse per row done)
  -> Signal dom Bool                    -- ^ logitsAllDone (all VocabularySize logits received)
  -> Signal dom Temperature             -- ^ temperature (0 = greedy argmax)
  -> Signal dom Seed                    -- ^ random seed
  -> (Signal dom Token, Signal dom Bool)  -- ^ (token, outputValid)
tokenSampler logitIdx logitValue logitValid logitsAllDone temperature seed =
  (tokenReg, validReg)
 where
  -- -------------------------------------------------------------------------
  -- State registers
  -- -------------------------------------------------------------------------
  state     = register SIdle             nextState
  scanCount = register (0 :: Unsigned 32) nextScanCount
  argIdx    = register (0 :: Token)       nextArgIdx
  argVal    = register (minBound :: FixedPoint) nextArgVal
  sumExpR   = register (0 :: FixedPoint)  nextSumExp
  randR     = register (0 :: FixedPoint)  nextRand
  cdfR      = register (0 :: FixedPoint)  nextCdf
  tokenReg  = register (0 :: Token)       nextToken
  validReg  = register False              nextValid

  -- -------------------------------------------------------------------------
  -- BRAM: VocabularySize × FixedPoint (32 bits × 512 = 2 KB for NANO)
  --   SFill    phase: write raw logits at their row index
  --   SExpScan phase: overwrite with exp values
  --   SCDFScan phase: read exp values for CDF walk
  -- 1-cycle read latency: rdAddr set in cycle N → bramOut valid in cycle N+1
  -- -------------------------------------------------------------------------
  bramOut :: Signal dom FixedPoint
  bramOut = blockRam (repeat (0 :: FixedPoint) :: Vec VocabularySize FixedPoint) bramRdAddr bramWrCmd

  -- BRAM read address: the index we want data for NEXT cycle
  bramRdAddr :: Signal dom (Index VocabularySize)
  bramRdAddr = fromIntegral <$> scanCount

  -- BRAM write: logits during fill; exp values during exp scan
  bramWrCmd :: Signal dom (Maybe (Index VocabularySize, FixedPoint))
  bramWrCmd =
    mux (inFillPhase .&&. logitValid)
        (Just <$> bundle (logitIdx, logitValue))
    $ mux (state .==. pure SExpScan .&&. scanCount .>. 0)
          -- scanCount-1 is the index whose data we received this cycle
          (Just <$> bundle (fromIntegral <$> (scanCount - 1), expVal))
    $ pure Nothing

  -- -------------------------------------------------------------------------
  -- Derived combinational values
  -- -------------------------------------------------------------------------
  vocSize :: Unsigned 32
  vocSize = natToNum @VocabularySize

  -- Prevent division by zero
  tempClamped :: Signal dom FixedPoint
  tempClamped = max 3.90625e-3 <$> temperature  -- BISECT: was (1/256)

  sumExpClamped :: Signal dom FixedPoint
  sumExpClamped = max 3.90625e-3 <$> sumExpR  -- BISECT: was (1/256)

  -- SExpScan: bramOut = logit[scanCount-1]; compute softmax numerator
  expVal :: Signal dom FixedPoint
  expVal = expF <$> ((bramOut - argVal) / tempClamped)

  -- SCDFScan: bramOut = exp[scanCount-1]; compute normalised probability mass
  probVal :: Signal dom FixedPoint
  probVal = bramOut / sumExpClamped

  -- Random uniform [0,1) sampled when logitsAllDone fires
  randUniform :: Signal dom FixedPoint
  randUniform = uniformRandom01Generator logitsAllDone seed

  -- True while we are (or transitioning into) the fill phase
  inFillPhase :: Signal dom Bool
  inFillPhase = (state .==. pure SFill)
             .||. (state .==. pure SIdle .&&. logitValid)

  -- A new argmax candidate arrived this cycle
  isNewBest :: Signal dom Bool
  isNewBest = inFillPhase .&&. logitValid .&&. (logitValue .>. argVal)

  -- Argmax including any update happening this cycle (avoids 1-cycle lag for
  -- the last logit arriving simultaneously with logitsAllDone)
  effectiveArgIdx :: Signal dom Token
  effectiveArgIdx = mux isNewBest (fromIntegral <$> logitIdx) argIdx

  -- CDF crossed the random draw threshold
  cdfHit :: Signal dom Bool
  cdfHit = state .==. pure SCDFScan
        .&&. scanCount .>. 0
        .&&. (cdfR + probVal .>=. randR)

  -- Exhausted all vocabulary items without hitting (floating-point rounding)
  cdfEnd :: Signal dom Bool
  cdfEnd = state .==. pure SCDFScan .&&. scanCount .==. pure vocSize

  -- Use argmax path when temperature is (effectively) zero
  useArgmax :: Signal dom Bool
  useArgmax = temperature .<. pure 3.90625e-3  -- BISECT STUB: was (1/256) to avoid SFixed division at synthesis time

  -- Fires in the cycle we decide on the final token
  justDone :: Signal dom Bool
  justDone = (state .==. pure SFill .&&. logitsAllDone .&&. useArgmax)
          .||. cdfHit
          .||. cdfEnd

  -- -------------------------------------------------------------------------
  -- Next-state
  -- -------------------------------------------------------------------------
  nextState =
    mux (state .==. pure SDone)                                       (pure SIdle)
    $ mux justDone                                                    (pure SDone)
    $ mux (state .==. pure SFill .&&. logitsAllDone .&&. (not <$> useArgmax))
                                                                      (pure SExpScan)
    $ mux (state .==. pure SExpScan .&&. scanCount .==. pure vocSize) (pure SCDFScan)
    $ mux inFillPhase                                                 (pure SFill)
    $ state

  -- -------------------------------------------------------------------------
  -- Scan counter (0 = BRAM warm-up cycle; 1..vocSize = active processing)
  -- -------------------------------------------------------------------------
  nextScanCount =
    mux (state .==. pure SFill .&&. logitsAllDone)                    (pure 0)
    $ mux (state .==. pure SExpScan .&&. scanCount .==. pure vocSize) (pure 0)
    $ mux (state .==. pure SExpScan .&&. scanCount .<. pure vocSize)  (scanCount + 1)
    $ mux (state .==. pure SCDFScan .&&. scanCount .<. pure vocSize
           .&&. (not <$> cdfHit))                                     (scanCount + 1)
    $ scanCount

  -- -------------------------------------------------------------------------
  -- Argmax registers
  -- -------------------------------------------------------------------------
  nextArgIdx = mux isNewBest (fromIntegral <$> logitIdx) argIdx

  nextArgVal =
    mux (state .==. pure SDone) (pure minBound)  -- reset for next token
    $ mux isNewBest logitValue
    $ argVal

  -- -------------------------------------------------------------------------
  -- Softmax accumulators
  -- -------------------------------------------------------------------------
  nextSumExp =
    mux (state .==. pure SExpScan .&&. scanCount .>. 0) (sumExpR + expVal)
    $ mux (state .==. pure SDone)                        (pure 0)
    $ sumExpR

  nextRand =
    mux (state .==. pure SFill .&&. logitsAllDone .&&. (not <$> useArgmax))
        randUniform
    $ randR

  nextCdf =
    mux (state .==. pure SCDFScan .&&. scanCount .>. 0) (cdfR + probVal)
    $ mux (state .==. pure SExpScan .&&. scanCount .==. pure vocSize)  (pure 0)
    $ cdfR

  -- -------------------------------------------------------------------------
  -- Output token
  -- -------------------------------------------------------------------------
  nextToken =
    -- Argmax path: use effective argmax (includes last logit if simultaneous)
    mux (state .==. pure SFill .&&. logitsAllDone .&&. useArgmax)
        effectiveArgIdx
    -- CDF hit: token is the index that tipped the CDF (scanCount-1 = Unsigned 32)
    $ mux cdfHit (scanCount - 1)
    -- CDF fallback: last index
    $ mux cdfEnd (pure (fromIntegral (maxBound :: Index VocabularySize)))
    $ tokenReg

  -- nextValid pulses True in the cycle after justDone (registers update)
  nextValid = justDone

-- ---------------------------------------------------------------------------
-- Synthesis wrapper
-- ---------------------------------------------------------------------------
{-# ANN samplerTop
  (Synthesize
    { t_name   = "token_sampler"
    , t_inputs =
        [ PortName "clk"
        , PortName "rst"
        , PortName "en"
        , PortName "logit_idx"
        , PortName "logit_value"
        , PortName "logit_valid"
        , PortName "logits_all_done"
        , PortName "temperature"
        , PortName "seed"
        ]
    , t_output = PortProduct ""
        [ PortName "token"
        , PortName "output_valid"
        ]
    }) #-}
samplerTop
  :: Clock System
  -> Reset System
  -> Enable System
  -> Signal System (Index VocabularySize)
  -> Signal System FixedPoint
  -> Signal System Bool
  -> Signal System Bool
  -> Signal System Temperature
  -> Signal System Seed
  -> (Signal System Token, Signal System Bool)
samplerTop = exposeClockResetEnable tokenSampler
