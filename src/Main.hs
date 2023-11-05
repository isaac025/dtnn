module Main where

import Control.Monad
import Control.Monad.Random
import Data.Kind
import Data.List
import Data.Maybe
import Data.Singletons
import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import System.Environment
import Text.Read

data Weights i o = W
    { wBiases :: !(R o)
    , wNodes :: !(L o i)
    }

data Network :: Nat -> [Nat] -> Nat -> Type where
    O :: !(Weights i o) -> Network i '[] o
    (:&~) :: (KnownNat h) => !(Weights i h) -> !(Network h hs o) -> Network i (h ': hs) o

infixr 5 :&~

randomWeights :: (MonadRandom m, KnownNat i, KnownNat o) => m (Weights i o)
randomWeights i o = do
    seed1 :: Int <- getRandom
    seed2 :: Int <- getRandom
    let wb = randomVector seed1 Uniform o * 2 - 1
        wn = uniformSample seed2 o (replicate i (-1, 1))
    pure $ W wb wn

randomNet :: forall m i hs o. (MonadRandom m, SingI hs, KnownNat i, KnownNat o) => m (Network i hs o)
randomNet = go sing
  where
    go :: forall h hs'. (KnownNat h) => Sing hs' -> m (Network h hs' o)
    go = \case
        SNil -> O <$> randomWeights
        SNat `SCons` ss -> (:&~) <$> randomWeights <*> go ss

logistic :: (Floating a) => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: (Floating a) => a -> a
logistic' x = logix * (1 - logix)
  where
    logix = logistic x

runLayer :: (KnownNat i, KnownNat o) => Weights i o -> R i -> R o
runLayer (W wb wn) v = wb + wn #> v

runNet :: (KnownNat i, KnownNat o) => Network i hs o -> R i -> R o
runNet = \case
    O w -> \(!v) -> logistic (runLayer w v)
    (w :&~ n') -> \(!v) ->
        let v' = logistic (runLayer w v)
         in runNet n' v'

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
    args <- getArgs
    let n = readMaybe =<< (args !!? 0)
        rate = readMaybe =<< (args !!? 1)
    putStrLn "Training network..."
    putStrLn
        =<< evalRandIO
            ( netTest
                (fromMaybe 0.25 rate)
                (fromMaybe 500000 n)
            )
  where
    (!!?) :: [a] -> Int -> Maybe a
    xs !!? i = listToMaybe (drop i xs)
