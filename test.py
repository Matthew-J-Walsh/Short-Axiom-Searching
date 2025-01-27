import subprocess
import sys

name_defs: dict[str, str] = {
    "BAAN": "baan",
    "BAC": "bac",
    "BACN": "bacn",
    "BAON": "baon",
    "CF": "cf",
    "CT": "ct",
    "CN": "cn",
    "HADN": "hadn",
    "LUK3VI": "luk3vi",
    "RAN": "ran"
}
for name, def_part in name_defs.items():
    subprocess.check_call([sys.executable, "run.py", name, "11", f"runs/{def_part}_definition.json", "--reset_progress"])