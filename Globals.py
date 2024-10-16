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
    """Default constant value"""
    predicate_orientation: bool | None
    """What value the predicate of this constant should be, if any"""

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
"""Definition of classical truth predicate"""
CLASSICAL_IMPLICATION = OperationSpec("C", "i", 2, numpy_read_only_array([[1, 1], [0, 1]]))
"""Definition of classical implication function"""
CLASSICAL_NEGATION = OperationSpec("N", "n", 1, numpy_read_only_array([1, 0]))
"""Definition of classical negation function"""
CLASSICAL_TRUE = ConstantSpec("T", "o", 1, True)
"""Definition of classical True constant"""
CLASSICAL_FALSE = ConstantSpec("F", "f", 0, False)
"""Definition of classical False constant"""
LUKASIEWICZ_TRUTH = OperationSpec("T", "t", 1, numpy_read_only_array([False, False, True], dtype=np.bool_))
"""Definition of truth in a Lukasiewicz 3-valued logic system"""
LUKASIEWICZ_IMPLICATION = OperationSpec("C", "i", 2, numpy_read_only_array([[2, 2, 2], [1, 2, 2], [0, 1, 2]]))
"""Definition of implication in a Lukasiewicz 3-valued logic system"""
LUKASIEWICZ_NEGATION = OperationSpec("N", "n", 1, numpy_read_only_array([2, 1, 0]))
"""Definition of negation in a Lukasiewicz 3-valued logic system"""

CN_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_TRUTH, CLASSICAL_IMPLICATION, CLASSICAL_NEGATION)
"""Operations for Propositional logic"""
CN_CONSTANTS: tuple[ConstantSpec, ...] = ()
"""Constants for Propositional logic"""
CN_SPEC: ModelSpec = ModelSpec(CN_OPERATIONS, CN_CONSTANTS)
"""Spec for Propositional logic"""
C_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_TRUTH, CLASSICAL_IMPLICATION)
"""Operations for Implicational logics"""
C0_CONSTANTS: tuple[ConstantSpec, ...] = (CLASSICAL_FALSE,)
"""Constants for Implication logic with False constant"""
C1_CONSTANTS: tuple[ConstantSpec, ...] = (CLASSICAL_TRUE,)
"""Constants for Implication logic with True constant"""
C0_SPEC: ModelSpec = ModelSpec(C_OPERATIONS, C0_CONSTANTS)
"""Spec for Implication logic with False Constant"""
C1_SPEC: ModelSpec = ModelSpec(C_OPERATIONS, C1_CONSTANTS)
"""Spec for Implication logic with True Constant"""
LUKASIEWICZ_3VI_OPERATIONS: tuple[OperationSpec, OperationSpec] = (LUKASIEWICZ_TRUTH, LUKASIEWICZ_IMPLICATION)
"""Operations for Lukasiewicz 3-valued logic system implicational fragment"""
LUKASIEWICZ_3VI_CONSTANTS: tuple[ConstantSpec, ...] = ()
"""Constants for Lukasiewicz 3-valued logic system implicational fragment"""
LUKASIEWICZ_3VI_SPEC: ModelSpec = ModelSpec(LUKASIEWICZ_3VI_OPERATIONS, LUKASIEWICZ_3VI_CONSTANTS)
"""Spec for Lukasiewicz 3-valued logic system implicational fragment"""


VERIFY_ALL_FORMULAS: bool = False
"""Should all formulas be checked (takes a very long time)"""


