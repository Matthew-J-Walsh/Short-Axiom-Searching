from __future__ import annotations
from typing import Literal

from Globals import *

from MathUtilities import bells

#Fill = tuple[int, int]
class Fill(NamedTuple):
    point: int
    size: int

class FillTable(NamedTuple):
    fills: np.ndarray
    subsumptive_cleaves: np.ndarray

    @staticmethod
    def get_fill_table(size: int) -> FillTable:
        #raise Warning("Depreciated")
        _initialize_fill_table(size)
        return FillTable(_fill_table_fills[:size, :bells(size)], _fill_table_subsumptive_table[:bells(size), :bells(size)])

def _fill_injection(A: np.ndarray[Any, np.dtype[np.int8]], B: np.ndarray[Any, np.dtype[np.int8]]) -> dict[int, int] | Literal[False]:
    mapping: dict[int, int] = {}
    for a, b in zip(A, B):
        if a in mapping.keys():
            if b!=mapping[a]:
                return False
        else:
            mapping[a] = b
    return mapping

def _generate_subsumptive_table_dumb(count: int, arr: np.ndarray) -> np.ndarray:
    subsumptive_table = np.zeros((arr.shape[0], arr.shape[0]), dtype=np.int8)

    subsumptive_table[0, 0] = 1
    for i, j in itertools.product(range(arr.shape[0]), repeat=2):
        subsumptive_table[i, j] = 1 if _fill_injection(arr[j], arr[i]) else 0
    
    return subsumptive_table

def _generate_subsumptive_table_pseudo(count: int, arr: np.ndarray) -> np.ndarray:
    #tab(i,j)==1 iff arr[i] being non-tautological implies arr[j] is non-tautological
    #contrapositively: tab(i,j)==1 iff arr[j] being tautological implies arr[i] is tautological
    #decided by: tab(i,j)==1 iff arr[j] has an injection to arr[i]
    #this lets us run through all forms iteratively, and whenever we find a form isn't tautological under the standard model we can cleave "downward"
    #then we run backward through all the forms with the counter-models, cleaving "upward"
    #When we find a form that is standardly-tautological and has no counter-models, we toss it to vampire
    #If vampire finds something, we add that counter-model, letting us cleave, otherwise we skip and add it to a file of "unclassified formula"
    splits = np.array([bells(i) for i in range(count+1)], dtype=np.int8)
    subsumptive_table = np.identity(arr.shape[0], dtype=np.int8)

    print(splits)
    subsumptive_table[0, 0] = 1
    for i in range(1, len(splits)-1):
        for j in range(splits[i], splits[i+1]-1):
            m = arr[j].max()
            #print(arr[j])
            #print(m)
            #print(arr[splits[m-1]-1])
            for k in range(splits[m-1]-1, splits[m]):
                if _fill_injection(arr[j], arr[k]):
                    subsumptive_table[k, j] = 1
                    subsumptive_table[:, j] += subsumptive_table[:, k]
        subsumptive_table[:splits[i+1]-1, splits[i+1]-1] = 1
    subsumptive_table[:splits[-1]-1, splits[-1]-1] = 1
    
    subsumptive_table = (subsumptive_table!=0).astype(int)

    return subsumptive_table

def _check_subsumptive_table(count: int, arr: np.ndarray) -> None:
    pseudo = _generate_subsumptive_table_pseudo(count, arr)
    dumb = _generate_subsumptive_table_dumb(count, arr)
    try:
        assert (pseudo == dumb).all()
    except:
        #raise AssertionError(np.argwhere(pseudo != dumb))
        for i, j in np.argwhere(pseudo != dumb):
            print("=============================")
            print(str(i)+";"+str(j))
            print(str(dumb[i, j])+";"+str(pseudo[i, j]))
            print("\t"+str(arr[j]))
            print("\t"+str(arr[i]))
        raise AssertionError()

def _add_with_carry(arr: np.ndarray) -> np.ndarray:
    if len(arr)==1:
        return np.array(arr)
    
    arr[-1] += 1
    if arr[-1] <= max(arr[:-1]) + 1:
        return arr
    else:
        return np.concatenate((_add_with_carry(arr[:-1]), np.array([0])))

_fill_table_fills: np.ndarray = np.zeros((0, 0))
_fill_table_subsumptive_table: np.ndarray = np.zeros((0, 0))

def _initialize_fill_table(size: int) -> None:
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

def fill_iterator(size: int) -> Iterable[Fill]:
    _initialize_fill_table(size)
    for i in range(bells(size)):
        yield Fill(i, size)

@functools.cache
def split_fill(fill: Fill, split_sizes: tuple[int, ...]) -> tuple[tuple[Fill, DimensionalReference], ...]:
    i: int = _fill_table_fills.shape[0] - fill.size
    assert i >= 0, str(fill)+" ... "+str(_fill_table_fills.shape)
    splitted: list[tuple[Fill, DimensionalReference]] = []
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
                splitted.append((Fill(j, s), tuple([injection[i] for i in range(s)])))
                break
        i += s
    assert len(splitted)==len(split_sizes)
    return tuple(splitted)

def fill_dimensions(fill: Fill) -> DimensionalReference:
    return _fill_table_fills[-fill.size:, fill.point].tolist()

def Cleaver(size: int) -> np.ndarray:
    return np.ones(bells(size), dtype=np.bool_)

def FullFill(size: int) -> Fill:
    _initialize_fill_table(size)
    return Fill(bells(size)-1, size)

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

def _point_to_fill(fill: np.ndarray) -> Fill:
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
    return Fill(_point_to_fill_cached(tuple(fixed_point)), fill.shape[0])

def fill_result_disassembly(evaluation: ModelArray) -> np.ndarray:
    falses = np.vstack(np.logical_not(evaluation).nonzero())
    cleaver = Cleaver(evaluation.ndim)
    for i in range(falses.shape[1]):
        cleaver *= fill_downward_cleave(_point_to_fill(falses[:, i]).point, evaluation.ndim)

    return cleaver


