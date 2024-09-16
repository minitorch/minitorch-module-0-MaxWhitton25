"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x, y):
    return x * y

def id(x):
    return x

def add(x, y):
    return x + y

def neg(x):
    return -x

def lt(x, y):
    return 1.0 if x < y else 0.0

def eq(x, y):
    return 1.0 if x == y else 0.0

def max(x, y):
    return x if x > y else y

def is_close(x, y):
    return abs(x - y) < 1e-2

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def relu(x):
    return x if x > 0 else 0.0

def log(x):
    return math.log(x)

def exp(x):
    return math.exp(x)

def log_back(x, d):
    return d / x

def inv(x):
    return 1.0 / x

def inv_back(x, d):
    return -d / (x * x)

def relu_back(x, d):
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    return [fn(x) for x in ls]

def zipWith(fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return [fn(x, y) for x, y in zip(ls1, ls2)]

def reduce(fn: Callable[[float, float], float], start: float, ls: Iterable[float]) -> float:
    result = start
    for x in ls:
        result = fn(result, x)
    return result

def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg, ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0, ls)

def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0, ls)
