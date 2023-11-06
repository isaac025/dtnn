module Main where

import Control.Monad
import Control.Monad.Random
import Data.Binary qualified as B
import Data.Kind
import Data.List
import Data.Maybe
import Data.Singletons
import GHC.Generics (Generic)
import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import System.Environment
import Text.Read

data Weights i o = W
    { wBiases :: !(R o)
    , wNodes :: !(L o i)
    }
    deriving (Show, Generic)

instance (KnownNat i, KnownNat o) => B.Binary (Weights i o)

data Network :: Nat -> [Nat] -> Nat -> Type where
    O :: !(Weights i o) -> Network i '[] o
    (:&~) :: (KnownNat h) => !(Weights i h) -> !(Network h hs o) -> Network i (h ': hs) o

infixr 5 :&~

instance (KnownNat i, SingI hs, KnownNat o) => B.Binary (Network i hs o) where
    put = putNet
    get = getNet

putNet :: (KnownNat i, KnownNat o) => Network i hs o -> B.Put
putNet = \case
    O w -> B.put w
    w :&~ n -> B.put w *> putNet n

getNet :: forall i hs o. (KnownNat i, KnownNat o) => Sing hs -> B.Get (Network i hs o)
getNet = \case
    SNil -> o <$> get
    SNat `SCons` ss -> (:&~) <$> B.get <*> getNet ss

data OpaqueNet :: Nat -> Nat -> Type where
    ONet :: Network i hs o -> OpaqueNet i o

putONet :: (KnownNat i, KnownNat o) => OpaqueNet i o -> B.Put
putONet (ONet net) = do
    B.put (hiddenStruct net)
    putNet net

getONet :: (KnownNat i, KnownNat o) => B.Get (OpaqueNet i o)
getONet = do
    hs <- B.get
    withSomeSing hs $ \ss -> ONet <$> getNet ss

instance (KnownNat i, KnownNat o) => B.Binary (OpaqueNet i o) where
    put = putONet
    get = getONet

type OpaqueNet' i o r = (forall hs. Network i hs o -> r) -> r

hiddenStruct :: Network i hs o -> [Integer]
hiddenStruct = \case
    O _ -> []
    _ :&~ (n' :: Network h hs' o) ->
        natVal (Proxy @h) : hiddenStruct n'

logistic :: (Floating a) => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: (Floating a) => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

randomWeights :: (MonadRandom m, KnownNat i, KnownNat o) => m (Weights i o)
randomWeights i o = do
    seed1 :: Int <- getRandom
    seed2 :: Int <- getRandom
    let wb = randomVector seed1 Uniform o * 2 - 1
        wn = uniformSample seed2 o (replicate i (-1, 1))
    pure $ W wb wn

randomNet :: forall m i hs o. (MonadRandom m, SingI hs, KnownNat i, KnownNat o) => m (Network i hs o)
randomNet = randomNet' sing

randomNet' :: forall m i hs o. (MonadRandom m, KnownNat i, KnownNat o) => Sing hs -> m (Network i hs o)
randomNet' = \case
    SNil -> O <$> randomWeights
    SNat `SCons` ss -> (:&~) <$> randomWeights <*> randomNet' ss

randomONet :: (MonadRandom m, KnownNat i, KnownNat o) => [Integer] -> m (OpaqueNet i o)
randomONet hs = case toSing hs of
    SomeSing ss -> ONet <$> randomNet' ss

runLayer :: (KnownNat i, KnownNat o) => Weights i o -> R i -> R o
runLayer (W wb wn) v = wb + wn #> v

runNet :: (KnownNat i, KnownNat o) => Network i hs o -> R i -> R o
runNet = \case
    O w -> \(!v) -> logistic (runLayer w v)
    (w :&~ n') -> \(!v) ->
        let v' = logistic (runLayer w v)
         in runNet n' v'

runOpaqueNet :: (KnownNat i, KnownNat o) => OpaqueNet i o -> R i -> R o
runOpaqueNet (ONet n) = runNet n

numHiddens :: OpaqueNet i o -> Int
numHiddens (ONet n) = go n
  where
    go :: Network i hs o -> Int
    go = \case
        O _ -> O
        _ :&~ n' -> 1 + go n'

runOpaqueNet' :: (KnownNat i, KnownNat o) => OpaqueNet' i o (R o) -> R i -> R o
runOpaqueNet' oN x = oN (\n -> runNet n x)

oNet' :: Network i hs o -> OpaqueNet' i o r
oNet' n = \f -> f n

withRandomONet' :: (MonadRandom m, KnownNat i, KnownNat o) => [Integer] -> (forall hs. Network i hs o -> m r) -> m r
withRandomONet' hs f =
    withSomeSing hs $ \ss -> do
        net <- randomNet' ss
        f net
numHiddens' :: OpaqueNet' i o Int -> Int
numHiddens' oN = oN go
  where
    go :: Network i hs o -> Int
    go = \case
        O _ -> 0
        _ :&~ n' -> 1 + go n'

main :: IO ()
main = do
    putStrLn "What hidden layer structure do you want?"
    hs <- readLn
    n <- randomONet hs
    case n of
        ONet (net :: Network 10 hs 3) -> do
            print net

main' :: IO ()
main' = do
    putStrLn "What hidden layer structure do you want?"
    hs <- readLn
    n <- randomONet hs
    withRandomONet' hs $ \(net :: Network 10 hs 3) -> do
        print net
