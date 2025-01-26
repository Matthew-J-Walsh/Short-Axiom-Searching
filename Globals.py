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
from collections import Counter, deque
from typing import Any
import warnings
import argparse
import json
#import psutil

#kernprof fix
try:
    profile # type: ignore
except NameError:
    def profile(func):
        return func

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
    associative: bool
    """Is this operation associative"""
    __hash__ = hash_tuple_with_ndarray

    @staticmethod
    def parse(inpt: dict) -> "OperationSpec":
        return OperationSpec(inpt["symbol"], inpt["tptp_symbol"], inpt["arity"], numpy_read_only_array(inpt["default_table"]), inpt["associative"])

class PredicateSpec(NamedTuple):
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
    associative: bool
    """Is this operation associative"""
    __hash__ = hash_tuple_with_ndarray

    @staticmethod
    def parse(inpt: dict) -> "PredicateSpec":
        return PredicateSpec(inpt["symbol"], inpt["tptp_symbol"], inpt["arity"], inpt["default_table"] if inpt["default_table"] is None else numpy_read_only_array(inpt["default_table"], dtype=np.bool_), inpt["formation_style"], inpt["associative"])

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

    @staticmethod
    def parse(inpt: dict) -> "ConstantSpec":
        return ConstantSpec(inpt["symbol"], inpt["tptp_symbol"], inpt["default_value"], inpt["predicate_orientation"])

class ModelSpec(NamedTuple):
    """Spec for a model operators and constants for loading from a file"""    
    prefix: PredicateSpec
    """Prefix operator"""
    operators: tuple[OperationSpec, ...]
    """Operators in order"""    
    constants: tuple[ConstantSpec, ...]
    """Constants in order"""    

class FunctionStackElement(NamedTuple):
    func: OperationSpec | PredicateSpec
    """Function being used"""
    rem_inpts: list[int]
    """Number of inputs needed, list so we can mutate"""
    inpt_tab: list[int]
    """Table of input values (so far)"""

def numpy_read_only_array(*args, dtype: type = np.int8) -> np.ndarray:
    """Make a numpy read only array

    Parameters
    ----------
    *args
        Iterable to make the array
    dtype : type, optional
        dtype to make the read only array, by default np.int8

    Returns
    -------
    np.ndarray
        Read only numpy array
    """    
    arr = np.array(*args, dtype=dtype)
    arr.setflags(write=False)
    return arr


