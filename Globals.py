import copy
from typing import Any, Literal, Union, Sequence, Iterable, Callable, NamedTuple, Protocol
import numpy as np
import scipy
import scipy.special
import itertools
import functools
import string
import os
import re
from io import TextIOWrapper
from datetime import datetime
from collections import Counter
from typing import Any

VARIABLE_SYMBOLS = string.ascii_lowercase
VAMPIRE_VARIABLE_SYMBOLS = string.ascii_uppercase

ModelArray = np.ndarray[Any, np.dtype[np.int8]]
DimensionalReference = tuple[int, ...]

def hash_tuple_with_ndarray(self) -> int:
    return hash(tuple(
        elem.tobytes() if isinstance(elem, np.ndarray) else elem for elem in self
    ))

class OperationSpec(NamedTuple):
    """Specification for a operation
    """    
    symbol: str
    """Symbol of the operation"""
    vampire_symbol: str
    """Symbol of the operation in vampire"""
    arity: int
    """Arity of the operation"""
    default_table: ModelArray
    """Default function table"""
    __hash__ = hash_tuple_with_ndarray

class ConstantSpec(NamedTuple):
    """Specification for a constant
    """    
    symbol: str
    """Symbol of the operation"""
    vampire_symbol: str
    """Symbol of the operation in vampire"""
    default_value: int
    __hash__ = hash_tuple_with_ndarray

class ModelSpec(NamedTuple):
    """Spec for a model operators and constants for loading from a file"""    
    operators: tuple[OperationSpec, ...]
    """Operators in order, first operator is unitary prefix"""    
    constants: tuple[ConstantSpec, ...]
    """Constants in order"""    

class FunctionStackElement(NamedTuple):
    func: OperationSpec
    """Function being used"""
    rem_inpts: list[int]
    """Number of inputs needed, list so we can mutate"""
    inpt_tab: list[int]
    """Table of input values (so far)"""

def numpy_read_only_array(*args, dtype: type = np.int8) -> np.ndarray:
    arr = np.array(*args, dtype=dtype)
    arr.setflags(write=False)
    return arr

CLASSICAL_TRUTH = OperationSpec("T", "t", 1, numpy_read_only_array([False, True], dtype=np.bool_))
CLASSICAL_IMPLICATION = OperationSpec("C", "i", 2, numpy_read_only_array([[1, 1], [0, 1]]))
CLASSICAL_NEGATION = OperationSpec("N", "n", 1, numpy_read_only_array([1, 0]))
CLASSICAL_TRUE = ConstantSpec("T", "o", 1)
CLASSICAL_FALSE = ConstantSpec("F", "o", 0)

CN_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_TRUTH, CLASSICAL_IMPLICATION, CLASSICAL_NEGATION)
CN_CONSTANTS: tuple[ConstantSpec, ...] = ()
CN_SPEC: ModelSpec = ModelSpec(CN_OPERATIONS, CN_CONSTANTS)
C_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_TRUTH, CLASSICAL_IMPLICATION)
C0_CONSTANTS: tuple[ConstantSpec, ...] = (CLASSICAL_FALSE,)
C1_CONSTANTS: tuple[ConstantSpec, ...] = (CLASSICAL_TRUE,)
C0_SPEC: ModelSpec = ModelSpec(C_OPERATIONS, C0_CONSTANTS)
C1_SPEC: ModelSpec = ModelSpec(C_OPERATIONS, C1_CONSTANTS)




