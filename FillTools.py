from __future__ import annotations
from typing import Any


from Globals import *

from MathUtilities import bells

#Fill = tuple[int, int]
class FillPointer(NamedTuple):
    """Holder for information on fills, importantly: hashable
    """    
    point: int
    """Fill cannonical number (the 10th fill of any size is the same except for 0s at the start)"""
    size: int
    """Size of the fill to use"""

class FillTable(NamedTuple):
    """Depreciated object that holds fills and cleaves. Undocumented
    """    
    fills: np.ndarray
    subsumptive_cleaves: np.ndarray

    @staticmethod
    def get_fill_table(size: int) -> FillTable:
        #raise Warning("Depreciated")
        _initialize_fill_table(size)
        return FillTable(_fill_table_fills[:size, :bells(size)], _fill_table_subsumptive_table[:bells(size), :bells(size)])

class CleavingMatrix:
    full_size: int
    """Full fill size targeted"""
    constant_count: int
    """Number of constants"""
    cleaves: dict[tuple[int, ...], np.ndarray[Any, np.dtype[np.bool_]]]
    """Mapping from """

    def __init__(self, full_size: int, constant_count: int, allow_degeneracy: bool = False, no_cleaves: bool = False) -> None:
        self.full_size = full_size
        self.constant_count = constant_count
        if not no_cleaves:
            self.cleaves = {}
            for comb in itertools.product(range(self.constant_count + 1), repeat=self.full_size):
                counts = np.zeros(self.constant_count + 1, dtype=np.int8)
                for i in comb:
                    counts[i] += 1
                if not allow_degeneracy and (counts == 0).any(): #degenerate
                    continue
                self.cleaves[comb] = self.base_cleaver(counts[0])

    @staticmethod
    def base_cleaver(size: int) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Generates a cleaver for a specific size of fill

        Parameters
        ----------
        size : int
            Size to make the cleaver for

        Returns
        -------
        np.ndarray
            Base cleaver (all 1s)
        """    
        return np.ones(bells(size), dtype=np.bool_)
    
    def __imul__(self, other: CleavingMatrix) -> CleavingMatrix:
        assert self.full_size == other.full_size
        assert self.constant_count == other.constant_count
        for k in other.cleaves.keys():
            if k in self.cleaves.keys():
                self.cleaves[k] *= other.cleaves[k]
            else:
                self.cleaves[k] = other.cleaves[k]

        return self
    

def _fill_injection(A: np.ndarray[Any, np.dtype[np.int8]], B: np.ndarray[Any, np.dtype[np.int8]]) -> dict[int, int] | Literal[False]:
    """Determines the mapping that turns A into B. 
    Returns false if one doesn't exist.

    Parameters
    ----------
    A : np.ndarray[Any, np.dtype[np.int8]]
        First fill
    B : np.ndarray[Any, np.dtype[np.int8]]
        First fill

    Returns
    -------
    dict[int, int] | Literal[False]
        Mapping if it exists
    """    
    mapping: dict[int, int] = {}
    for a, b in zip(A, B):
        if a in mapping.keys():
            if b!=mapping[a]:
                return False
        else:
            mapping[a] = b
    return mapping

def _generate_subsumptive_table_dumb(size: int, arr: np.ndarray) -> np.ndarray:
    """Generates a subsumptive table for an array of fills by injection checking

    Parameters
    ----------
    size : int
        Size of the fills
    arr : np.ndarray
        Array holding the fills

    Returns
    -------
    np.ndarray
        Table indicating which fills subsume which other fills
    """    
    subsumptive_table = np.zeros((arr.shape[0], arr.shape[0]), dtype=np.int8)

    subsumptive_table[0, 0] = 1
    for i, j in itertools.product(range(arr.shape[0]), repeat=2):
        subsumptive_table[i, j] = 1 if _fill_injection(arr[j], arr[i]) else 0
    
    return subsumptive_table

def _add_with_carry(arr: np.ndarray) -> np.ndarray:
    """Iterative function for fills. Adds 1 with carry. Recursive
    Going left to right fills must be no greater than the max so far + 1, 
    this adds 1 to the left side and then carrys it over until that is the case.

    Parameters
    ----------
    arr : np.ndarray
        Array to add 1 to

    Returns
    -------
    np.ndarray
        Array with 1 added to it
    """    
    if len(arr)==1:
        return np.array(arr)
    
    arr[-1] += 1
    if arr[-1] <= max(arr[:-1]) + 1:
        return arr
    else:
        return np.concatenate((_add_with_carry(arr[:-1]), np.array([0])))

_fill_table_fills: np.ndarray = np.zeros((0, 0))
"""Holder for the current largest fill table"""
_fill_table_subsumptive_table: np.ndarray = np.zeros((0, 0))
"""Holder for the current largest subsumptive table"""

def _initialize_fill_table(size: int) -> None:
    """Makes sure that _fill_table_fills, the largest fill table, is atleast size large

    Parameters
    ----------
    size : int
        Size to make sure table can handle
    """    
    global _fill_table_fills, _fill_table_subsumptive_table
    if _fill_table_fills.shape[0] < size:
        fills = np.zeros((bells(size), size), dtype=np.int8)
        current = fills[0]
        for i in range(1, bells(size)):
            current = _add_with_carry(current.copy())
            fills[i] = current
        
        #fills = fills + 1
        subsumptive_table = (1 - _generate_subsumptive_table_dumb(size, fills)).astype(np.bool_) #subtact 1 so that 0s are on cleaved values
        fills = fills.T
        fills.setflags(write=False)
        subsumptive_table.setflags(write=False)
        
        _fill_table_fills = fills
        _fill_table_subsumptive_table = subsumptive_table

def fill_iterator(size: int) -> Iterable[FillPointer]:
    """Iterates through every FillPointer of a specific size

    Parameters
    ----------
    size : int
        Size to make fills for

    Yields
    ------
    Fill
        Fills of that size
    """    
    _initialize_fill_table(size)
    for i in range(bells(size)):
        yield FillPointer(i, size)

@functools.cache
def split_fill(fill: FillPointer, split_sizes: tuple[int, ...]) -> tuple[tuple[FillPointer, DimensionalReference], ...]:
    """Splits a fill

    Parameters
    ----------
    fill : Fill
        Fill to be split
    split_sizes : tuple[int, ...]
        Block sizes to split into, must sum to fill.size

    Returns
    -------
    tuple[tuple[Fill, DimensionalReference], ...]
        Tuple to utilize caching.
        Each (sub)tuple is a subfill and the reverse reference to re-calculate its original fill variables
    """    
    i: int = _fill_table_fills.shape[0] - fill.size
    assert i >= 0, str(fill)+" ... "+str(_fill_table_fills.shape)
    splitted: list[tuple[FillPointer, DimensionalReference]] = []
    for s in split_sizes:
        split_fill: np.ndarray[Any, np.dtype[np.int8]] = _fill_table_fills[i:i+s, fill.point]
        newfill: np.ndarray[Any, np.dtype[np.int8]] = split_fill - split_fill[0]
        injection: dict[int, int] | Literal[False] = _fill_injection(newfill, split_fill)
        assert injection
        #print(split_fill)
        #print(newfill)
        #print(injection)
        for j in range(_fill_table_fills.shape[1]):
            if (_fill_table_fills[-s:, j] == newfill).all():
                splitted.append((FillPointer(j, s), tuple([injection[i] for i in range(s)])))
                break
        i += s
    assert len(splitted)==len(split_sizes)
    return tuple(splitted)

def get_fill(fill: FillPointer) -> DimensionalReference:
    """Fill induced by a pointer

    Parameters
    ----------
    fill : FillPointer
        FillPointer to get the fill of

    Returns
    -------
    DimensionalReference
        List of ints of the dimension of each fill value, classically, a fill.
    """    
    return _fill_table_fills[-fill.size:, fill.point].tolist()

def full_fill(size: int) -> FillPointer:
    """Creates the most general fill of a size

    Parameters
    ----------
    size : int
        Size of fill to make

    Returns
    -------
    FillPointer
        Pointer to the fill
    """    
    _initialize_fill_table(size)
    return FillPointer(bells(size)-1, size)

def fill_downward_cleave(i: int, size: int) -> np.ndarray:
    #Cleave from non-tautological discovery at index i
    #Returns 0 on cleaved elements
    assert i < _fill_table_subsumptive_table.shape[0], str(i) + ", " + str(_fill_table_subsumptive_table.shape)
    return _fill_table_subsumptive_table[i, :bells(size)]

def fill_upward_cleave(i: int, size: int) -> np.ndarray:
    #Cleave from tautological discovery at index i
    #Returns 0 on cleaved elements
    assert i < _fill_table_subsumptive_table.shape[1]
    return _fill_table_subsumptive_table[:bells(size), i]

def Fillin(expression: np.ndarray, arr: np.ndarray) -> np.ndarray:
    assert len(expression.shape)==1, expression.shape
    assert ((expression==0).sum()==arr.shape[0]).all(), str((expression==0).sum())+", "+str(arr.shape[0])
    result = np.tile(expression.reshape(-1, 1), arr.shape[1])
    result[expression==0] = arr
    return result.T

@functools.cache
def _point_to_fill_cached(fill: tuple[int, ...]) -> int:
    point_arr = np.array(fill)
    for i in range(bells(point_arr.shape[0])):
        if (_fill_table_fills[-point_arr.shape[0]:, i] == point_arr).all():
            return i
    raise RuntimeError("Unabled find row for fill "+str(fill)+" possibly not normalized.")

def _point_to_fill(fill: np.ndarray) -> FillPointer:
    first_val: int = fill[0]
    cut: int = 0
    while cut + 1 < fill.shape[0] and fill[cut + 1] == first_val:
        cut += 1
    fill = fill[cut:]

    fixed_point: list[int] = []
    conversion: dict[int, int] = {}
    next_val = 0
    for i in range(fill.shape[0]):
        if not fill[i] in conversion.keys():
            conversion[fill[i]] = next_val
            next_val += 1
        fixed_point.append(conversion[fill[i]])
    return FillPointer(_point_to_fill_cached(tuple(fixed_point)), fill.shape[0])

def fill_result_disassembly_application(evaluation: ModelArray, constants: Sequence[int], cleave_direction: Literal["Upward"] | Literal["Downward"]) -> CleavingMatrix:
    fill_pairings: dict[tuple[int, ...], list[FillPointer]] = fill_disassembly_specified_fill_pairings(evaluation, constants)
    cleavematrix: CleavingMatrix = CleavingMatrix(evaluation.ndim, len(constants), no_cleaves = True)
    cleavematrix.cleaves = {}

    for constant_specifier, fills in fill_pairings.items():
        var_count: int = constant_specifier.count(0)
        cleaver: np.ndarray[Any, np.dtype[np.bool_]] = CleavingMatrix.base_cleaver(var_count)
        sorted_fills: list[FillPointer] = sorted(fills, key = lambda f: f.point, reverse = cleave_direction == "Upward")
        for fill in sorted_fills:
            if cleaver[fill.point]!=0:
                cleaver *= fill_downward_cleave(fill.point, var_count)
        if cleave_direction == "Upward":
            cleaver = np.logical_not(cleaver)
        cleavematrix.cleaves[constant_specifier] = cleaver

    return cleavematrix

def fill_disassembly_specified_fill_pairings(evaluation: ModelArray, constants: Sequence[int]) -> dict[tuple[int, ...], list[FillPointer]]:
    fill_pairings: dict[tuple[int, ...], list[FillPointer]] = {}
    falses = np.vstack(np.logical_not(evaluation).nonzero())
    for i in range(falses.shape[1]):
        #point = falses[:, i]
        constant_possibilities = [[0] + [k for k, c in enumerate(constants) if c==falses[j, i]] for j in range(evaluation.ndim)]
        for constant_combination in itertools.product(*constant_possibilities):
            #var_count: int = constant_combination.count(0)
            reduced_point: np.ndarray[Any, np.dtype[np.int8]] = np.array([pv for ccv, pv in zip(constant_combination, falses[:, i]) if ccv==0], dtype=np.int8)
            if constant_combination in fill_pairings.keys():
                fill_pairings[constant_combination].append(_point_to_fill(reduced_point))
            else:
                fill_pairings[constant_combination] = [_point_to_fill(reduced_point)]
    
    return fill_pairings
