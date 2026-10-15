-- Typeclasses in Haskell -- ad-hoc polymorphism done principled, on top of a
-- full Hindley-Milner type system where almost nothing is annotated.
--
-- A typeclass is an *interface with laws*: `class Eq a` declares that a type a
-- supports (==); an `instance Eq Color` supplies it. The compiler passes the
-- right implementation as a hidden "dictionary" argument -- so `(==)` is one
-- name with a different body per type, resolved at compile time from the
-- inferred types. This is the academic ancestor of Rust traits and Swift
-- protocols, and the cleaner cousin of Go/Java interfaces.
--
-- Everything below typechecks with ZERO type signatures required; the ones
-- present are documentation, not necessity (HM infers the most general type).
--
--   Run:    runghc typeclasses.hs       (or: ghc typeclasses.hs && ./typeclasses)
--   Check:  ghc -fno-code typeclasses.hs

module Main where

import Data.List (sortBy)

-- --------------------------------------------------------------------------- --
-- 1. Parametric polymorphism: types with variables, one implementation        --
-- --------------------------------------------------------------------------- --
-- `a` is a type variable; identity works for every a. Parametricity (Wadler's
-- "Theorems for Free") says the type `a -> a` *forces* this to be the identity.

identity :: a -> a
identity x = x

-- A polymorphic data type: a binary tree holding any element type.
data Tree a = Leaf | Node (Tree a) a (Tree a)

insert :: Ord a => a -> Tree a -> Tree a   -- Ord a: a bounded constraint
insert x Leaf = Node Leaf x Leaf
insert x t@(Node l v r)
  | x < v     = Node (insert x l) v r
  | x > v     = Node l v (insert x r)
  | otherwise = t

toList :: Tree a -> [a]
toList Leaf         = []
toList (Node l v r) = toList l ++ [v] ++ toList r

-- --------------------------------------------------------------------------- --
-- 2. Ad-hoc polymorphism: define a typeclass and instances                     --
-- --------------------------------------------------------------------------- --
-- `Shape` is our own class. Each instance gives a different `area` -- the same
-- dispatch a Rust `impl Trait` or a Go method set provides, but inferred.

class Shape a where
  area :: a -> Double
  describe :: a -> String
  describe s = "a shape of area " ++ show (area s)  -- default method

data Circle = Circle Double
data Rectangle = Rectangle Double Double

instance Shape Circle where
  area (Circle r) = pi * r * r

instance Shape Rectangle where
  area (Rectangle w h) = w * h
  describe (Rectangle w h) = "a " ++ show w ++ "x" ++ show h ++ " rectangle"

-- A function polymorphic over ANY Shape: bounded quantification via the
-- `Shape a =>` constraint. The dictionary for `area` is passed implicitly.
totalArea :: Shape a => [a] -> Double
totalArea = sum . map area

-- --------------------------------------------------------------------------- --
-- 3. Higher-kinded polymorphism: Functor abstracts over type CONSTRUCTORS      --
-- --------------------------------------------------------------------------- --
-- Functor is parameterized by `f :: * -> *` -- a type that still needs an
-- argument (like Tree, Maybe, []). This "abstraction over things that take a
-- type" is higher-kinded polymorphism, which Go and (pre-GAT) Rust cannot
-- express directly. `fmap` maps a function over whatever f holds.

instance Functor Tree where
  fmap _ Leaf         = Leaf
  fmap g (Node l v r) = Node (fmap g l) (g v) (fmap g r)

-- --------------------------------------------------------------------------- --
-- 4. Deriving: free instances of standard classes                             --
-- --------------------------------------------------------------------------- --
-- `deriving` asks the compiler to synthesize lawful Eq/Ord/Show instances.

data Color = Red | Green | Blue
  deriving (Eq, Ord, Show, Enum, Bounded)

allColors :: [Color]
allColors = [minBound .. maxBound]   -- uses the derived Bounded + Enum

main :: IO ()
main = do
  putStrLn "1. Parametric: a BST polymorphic in its element type"
  let nums = foldr insert Leaf [5, 3, 8, 1, 4, 7, 9 :: Int]
  putStrLn $ "   inorder(Tree Int) = " ++ show (toList nums)
  let words' = foldr insert Leaf ["pear", "apple", "fig", "kiwi"]
  putStrLn $ "   inorder(Tree String) = " ++ show (toList words')

  putStrLn "\n2. Ad-hoc: Shape instances dispatch area/describe per type"
  putStrLn $ "   " ++ describe (Circle 2)
  putStrLn $ "   " ++ describe (Rectangle 3 4)
  putStrLn $ "   totalArea [Circle 1, Circle 2] = "
    ++ show (totalArea [Circle 1, Circle 2])

  putStrLn "\n3. Higher-kinded: fmap (*10) over a Tree via Functor"
  putStrLn $ "   " ++ show (toList (fmap (* 10) nums))

  putStrLn "\n4. Deriving: Ord/Show/Enum for free"
  putStrLn $ "   sorted colors = "
    ++ show (sortBy compare [Blue, Red, Green])
  putStrLn $ "   all colors    = " ++ show allColors

  putStrLn "\n5. identity is forced by its type a -> a (parametricity)"
  putStrLn $ "   identity 42 = " ++ show (identity (42 :: Int))

  putStrLn "\nOK"
