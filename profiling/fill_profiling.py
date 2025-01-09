import sys
sys.path.insert(0, '../')
from memory_profiler import memory_usage
from FillTools import _initialize_fill_table

def run(target):
    def wrapper():
        _initialize_fill_table(target)
    
    max_mem = memory_usage(wrapper, max_usage=True) # type: ignore
    print(f"Memory usage for input {target} is {max_mem} MiB")

if __name__ == "__main__":
    target = int(sys.argv[1])
    run(target)