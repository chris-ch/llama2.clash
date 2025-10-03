module Model.Layers.Attention.AttentionHead
  ( attendHead ) where

import Clash.Prelude
import Model.Config (HeadDimension, SequenceLength)
import Model.Numeric.Types (FixedPoint)
import Model.Numeric.Fixed (expF)

attendHead
  :: Vec HeadDimension FixedPoint                      -- Q
  -> Vec SequenceLength (Vec HeadDimension FixedPoint) -- K rows
  -> Vec SequenceLength (Vec HeadDimension FixedPoint) -- V rows
  -> Index SequenceLength                     -- pos (inclusive)
  -> Vec HeadDimension FixedPoint
attendHead q ks vs pos =
  let
    scale :: FixedPoint
    scale = realToFrac (1.0 / sqrt ((natToNum @HeadDimension) :: Double))

    negBig :: FixedPoint
    negBig = minBound

    scores :: Vec SequenceLength FixedPoint
    scores =
      imap (\t krow ->
              let s = sum (zipWith (*) q krow) * scale
              in if fromEnum t <= fromEnum pos then s else negBig)
           ks

    m   = maximum scores
    exps = map (\s -> if s == negBig then 0 else expF (s - m)) scores
    d   = fold (+) exps
    probs = if d == 0 then repeat 0 else map (/ d) exps

    out  = foldl
             (\acc (p, vrow) -> zipWith (+) acc (map (* p) vrow))
             (repeat 0)
             (zip probs vs)
  in out
