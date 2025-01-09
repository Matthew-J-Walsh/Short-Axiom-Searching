import copy
from typing import Any, Literal, Union, Sequence, Iterable, Callable, NamedTuple, Protocol
import numpy as np
from scipy import sparse as sp
import scipy
import scipy.special
import itertools
import functools
import string
import os
import re
import time
import subprocess
from pathlib import Path
from io import TextIOWrapper
from datetime import datetime
from collections import Counter
from typing import Any
import warnings
import argparse
#import psutil

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
    tptp_symbol: str
    """Symbol of the operation in tptp"""
    arity: int
    """Arity of the operation"""
    default_table: ModelArray
    """Default function table"""
    __hash__ = hash_tuple_with_ndarray

class PrefixSpec(NamedTuple):
    """Specification for a operation
    """    
    symbol: str
    """Symbol of the operation"""
    tptp_symbol: str
    """Symbol of the operation in tptp"""
    arity: int
    """Arity of the operation"""
    default_table: ModelArray | None
    """Default function table or none if no default"""
    formation_style: str
    """How formulas are formed around this prefix, polish for 'F(x)', infix for 'x=x'"""
    mirrored: bool
    """Is this operation mirroring"""
    __hash__ = hash_tuple_with_ndarray

class ConstantSpec(NamedTuple):
    """Specification for a constant
    """    
    symbol: str
    """Symbol of the operation"""
    tptp_symbol: str
    """Symbol of the operation in tptp"""
    default_value: int
    """Default constant value"""
    predicate_orientation: bool | None
    """What value the predicate of this constant should be, if any"""

class ModelSpec(NamedTuple):
    """Spec for a model operators and constants for loading from a file"""    
    prefix: PrefixSpec
    """Prefix operator"""
    operators: tuple[OperationSpec, ...]
    """Operators in order"""    
    constants: tuple[ConstantSpec, ...]
    """Constants in order"""    

class FunctionStackElement(NamedTuple):
    func: OperationSpec | PrefixSpec
    """Function being used"""
    rem_inpts: list[int]
    """Number of inputs needed, list so we can mutate"""
    inpt_tab: list[int]
    """Table of input values (so far)"""

def numpy_read_only_array(*args, dtype: type = np.int8) -> np.ndarray:
    arr = np.array(*args, dtype=dtype)
    arr.setflags(write=False)
    return arr

CLASSICAL_TRUTH = PrefixSpec("T", "t", 1, numpy_read_only_array([False, True], dtype=np.bool_), "Polish", False)
"""Definition of classical truth predicate"""
CLASSICAL_IMPLICATION = OperationSpec("C", "i", 2, numpy_read_only_array([[1, 1], [0, 1]]))
"""Definition of classical implication function"""
CLASSICAL_NEGATION = OperationSpec("N", "n", 1, numpy_read_only_array([1, 0]))
"""Definition of classical negation function"""
CLASSICAL_DISJUNCTION = OperationSpec("D", "o", 2, numpy_read_only_array([[0, 1], [1, 1]]))
"""Definition of classical or operator"""
CLASSICAL_CONJUNCTION = OperationSpec("C", "a", 2, numpy_read_only_array([[0, 0], [0, 1]]))
"""Definition of classical and operator"""
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
PURE_EQUALITY = PrefixSpec("=", "=", 2, None, "Infix", True)
"""Definition of standard mathematical equality. To be equal they must be the same"""

CN_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_IMPLICATION, CLASSICAL_NEGATION)
"""Operations for Propositional logic"""
DN_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_DISJUNCTION, CLASSICAL_NEGATION)
"""Operations for disjunction + negation logics"""
AN_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_CONJUNCTION, CLASSICAL_NEGATION)
"""Operations for conjunction + negation logics"""
NO_CONSTANTS: tuple[ConstantSpec, ...] = ()
"""Empty constant list"""
CN_SPEC: ModelSpec = ModelSpec(CLASSICAL_TRUTH, CN_OPERATIONS, NO_CONSTANTS)
"""Spec for Propositional logic"""
C_OPERATIONS: tuple[OperationSpec, ...] = (CLASSICAL_IMPLICATION,)
"""Operations for Implicational logics"""
C0_CONSTANTS: tuple[ConstantSpec, ...] = (CLASSICAL_FALSE,)
"""Constants for Implication logic with False constant"""
C1_CONSTANTS: tuple[ConstantSpec, ...] = (CLASSICAL_TRUE,)
"""Constants for Implication logic with True constant"""
C0_SPEC: ModelSpec = ModelSpec(CLASSICAL_TRUTH, C_OPERATIONS, C0_CONSTANTS)
"""Spec for Implication logic with False Constant"""
C1_SPEC: ModelSpec = ModelSpec(CLASSICAL_TRUTH, C_OPERATIONS, C1_CONSTANTS)
"""Spec for Implication logic with True Constant"""
LUKASIEWICZ_3VI_OPERATIONS: tuple[OperationSpec, ...] = (LUKASIEWICZ_TRUTH, LUKASIEWICZ_IMPLICATION)
"""Operations for Lukasiewicz 3-valued logic system implicational fragment"""
LUKASIEWICZ_3VI_CONSTANTS: tuple[ConstantSpec, ...] = ()
"""Constants for Lukasiewicz 3-valued logic system implicational fragment"""
LUKASIEWICZ_3VI_SPEC: ModelSpec = ModelSpec(CLASSICAL_TRUTH, LUKASIEWICZ_3VI_OPERATIONS, LUKASIEWICZ_3VI_CONSTANTS)
"""Spec for Lukasiewicz 3-valued logic system implicational fragment"""
BOOLEAN_ALGEBRA_CN_SPEC: ModelSpec = ModelSpec(PURE_EQUALITY, CN_OPERATIONS, NO_CONSTANTS)
"""Spec for boolean algebra with implication and negation operators"""
BOOLEAN_ALGEBRA_DN_SPEC: ModelSpec = ModelSpec(PURE_EQUALITY, DN_OPERATIONS, NO_CONSTANTS)
"""Spec for boolean algebra with disjunction and negation operators"""
BOOLEAN_ALGEBRA_AN_SPEC: ModelSpec = ModelSpec(PURE_EQUALITY, AN_OPERATIONS, NO_CONSTANTS)
"""Spec for boolean algebra with conjunction and negation operators"""
BOOLEAN_ALGEBRA_CF_SPEC: ModelSpec = ModelSpec(PURE_EQUALITY, C_OPERATIONS, C0_CONSTANTS)
"""Spec for boolean algebra with implication and falsum"""
BOOLEAN_ALGEBRA_C_SPEC: ModelSpec = ModelSpec(PURE_EQUALITY, C_OPERATIONS, NO_CONSTANTS)
"""Spec for boolean algebra with just implication"""

def get_spec_reference(name: str) -> ModelSpec:
    match name:
        case "BAC":
            return BOOLEAN_ALGEBRA_C_SPEC
        case _:
            raise ValueError("Unknown spec: "+name)


VERIFY_ALL_FORMULAS: bool = False
"""Should all formulas be checked (takes a very long time)"""


