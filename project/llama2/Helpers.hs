module Helpers (
  liftA4
  , liftA5
) where

import Clash.Prelude

liftA4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = liftA3 f a b c <*> d

liftA5 :: Applicative g => (a -> b -> c -> d -> e -> f) -> g a -> g b -> g c -> g d -> g e -> g f
liftA5 f fa fb fc fd fe = f <$> fa <*> fb <*> fc <*> fd <*> fe
