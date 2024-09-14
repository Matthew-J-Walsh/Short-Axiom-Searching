from __future__ import annotations
from Globals import *

import gc
import sys
import types
from typing import TypeAlias

def memory_usage_of_class(cls: type) -> int:
    #kinda bad, doesn't count stuff like cached arrays
    size = 0
    for obj in gc.get_objects():
        if isinstance(obj, cls):
            size += sys.getsizeof(obj)
    return size

def total_memory() -> int:
    size = 0 
    for obj in gc.get_objects():
        try:
            size += sys.getsizeof(obj)
        except:
            continue
    return size

