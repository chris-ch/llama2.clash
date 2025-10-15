module LLaMa2.Helpers (
  liftA4, liftA5, liftA6
) where

import Clash.Prelude

liftA4 :: Applicative h => (a -> b -> c -> d -> e) -> h a -> h b -> h c -> h d -> h e
liftA4 h a b c d = liftA3 h a b c <*> d

liftA5 :: Applicative h => (a -> b -> c -> d -> e -> f) -> h a -> h b -> h c -> h d -> h e -> h f
liftA5 h a b c d e = liftA4 h a b c d <*> e

liftA6 :: Applicative h => (a -> b -> c -> d -> e -> f -> g) -> h a -> h b -> h c -> h d -> h e -> h f -> h g
liftA6 h a b c d e f = liftA5 h a b c d e <*> f
