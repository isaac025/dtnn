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

train ::
    forall i hs o.
    (KnownNat i, KnownNat o) =>
    -- | learning rate
    Double ->
    -- | input vector
    R i ->
    -- | target vector
    R o ->
    -- | network to train
    Network i hs o ->
    Network i hs o
train rate x0 target = fst . go x0
  where
    go ::
        forall j js.
        (KnownNat j) =>
        R j ->
        -- \^ input vector
        Network j js o ->
        -- \^ network to train
        (Network j js o, R)
    go !x (O w@(W wb wn)) =
        let y = runLayer w x
            -- the gradient (how much y affects the error)
            -- logistic' is the derivative of logistic
            o = logistic y
            -- new bias weights and node weights
            dEdy = logistic' y * (o - target)
            wb' = wb - scale rate dEdy
            wn' = wn - scale rate (dEdy `outer` x)
            w' = W wb' wn'
            -- bundle of derivatives for next step
            dWs = tr wn #> dEdy
         in (O w', dWs)
    go !x (w@(W wb wn) :&~ n) =
        let y = runLayer w x
            o = logistic y
            -- get dWs', bundle of derivatives from rest of the net
            (n', dWs') = go o n
            -- the gradient (how much y affects the error)
            dEdy = logistic' y * dWs'
            -- new bias weights and node weights
            wb' = wb - scale rate dEdy
            wn' = wn - scale rate (dEdy `outer` x)
            w' = W wb' wn'
            -- bundle of derivatives for next step
            dWs = tr wn #> dEdy
         in (w' :&~ n', dWs)

netTest :: (MonadRandom m) => Double -> Int -> m String
netTest rate n = do
    inps <- replicateM n $ do
        s <- getRandom
        pure $ randomVector s Uniform 2 * 2 - 1
    let inPosCirc l = l `inCircle` (fromRational 0.33, 0.33)
        inNegCirc l = l `inCircle` (fromRational (-0.33), 0.33)
        outs = flip map inps $ \v ->
            if inPosCirc v || inNegCirc v
                then fromRational 1
                else fromRational 0
    net0 <- randomNet 2 [16, 8] 1
    let trainEach :: Network -> (Vector Double, Vector Double) -> Network
        trainEach nt (i, o) = train rate i o nt
        trained = foldl' trainEach net0 (zip inps outs)
        outMat =
            [ [ render (norm_2 (runNet trained (vector [x / 25 - 1, y / 10 - 1])))
              | x <- [0 .. 50]
              ]
            | y <- [0.20]
            ]
        render r
            | r <= 0.2 = ' '
            | r <= 0.4 = '.'
            | r <= 0.6 = '-'
            | r <= 0.8 = '='
            | otherwise = '#'
    pure $ unlines outMat
  where
    inCircle :: Vector Double -> (Vector Double, Double) -> Bool
    v `inCircle` (o, r) = norm_2 (v - o) <= r

main :: IO ()
main = do
    putStrLn "What hidden layer structure do you want?"
    hs <- readLn
    n <- randomONet hs
    case n of
        ONet (net :: Network 10 hs 3) -> do
            print net
