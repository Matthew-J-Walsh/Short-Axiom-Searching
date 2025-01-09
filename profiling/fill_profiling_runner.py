import subprocess
from memory_profiler import memory_usage

targets = range(2, 12)

for target in targets:
    print(f"Profiling size: {target}")
    result = subprocess.run(["python", "fill_profiling.py", str(target)], 
                             capture_output=True, text=True)
    
    print(f"Result: {result.stdout}")
    print(f"Errors: {result.stderr}")