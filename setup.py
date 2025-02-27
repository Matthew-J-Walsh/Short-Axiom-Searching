import os
import sys
import stat
import subprocess
import urllib.request
import shutil
import argparse
from setuptools import setup
from setuptools.command.install import install

parser = argparse.ArgumentParser(description="Setup for Short Axiom Searching")

parser.add_argument("--skip_vampire", action='store_true', help="Skip installing vampire")
parser.add_argument("--skip_prover9", action='store_true', help="Skip installing prover9")
parser.add_argument("--include_libraries", action='store_true', help="Including installing and updating numpy and scipy")

args = parser.parse_args()

VAMPIRE_BINARY_URL = "https://github.com/vprover/vampire/releases/download/v4.9casc2024/vampire"
VAMPIRE_TARGET_PATH = "theorem_provers/vampire"
PROVER9_REPO_URL = "https://github.com/ai4reason/Prover9"
PROVER9_CLONE_PATH = "build/Prover9"
PROVER9_TARGET_PATH = "theorem_provers/prover9"

if not args.skip_vampire and not os.path.exists(VAMPIRE_TARGET_PATH):
    vampire_dir = os.path.dirname(VAMPIRE_TARGET_PATH)
    if not os.path.exists(vampire_dir):
        os.makedirs(vampire_dir)

    print(f"Downloading vampire binary from {VAMPIRE_BINARY_URL}")
    urllib.request.urlretrieve(VAMPIRE_BINARY_URL, VAMPIRE_TARGET_PATH)

    print(f"Setting executable permissions for {VAMPIRE_TARGET_PATH}")
    os.chmod(VAMPIRE_TARGET_PATH, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)

    print("Vampire fully setup.")

if not args.skip_prover9 and not os.path.exists(PROVER9_TARGET_PATH):
    print(f"Cloning Prover9 to {PROVER9_CLONE_PATH}")
    if not os.path.exists(PROVER9_CLONE_PATH):
        subprocess.check_call(["git", "clone", PROVER9_REPO_URL, PROVER9_CLONE_PATH])

    print("Building Prover9")
    subprocess.check_call(["make", "all"], cwd=PROVER9_CLONE_PATH)

    compiled_prover9_path = os.path.join(PROVER9_CLONE_PATH, "bin", "prover9")
    assert os.path.exists(compiled_prover9_path), "Prover9 build failure"
    print(f"Moving Prover9 binary to {PROVER9_TARGET_PATH}")
    shutil.move(compiled_prover9_path, PROVER9_TARGET_PATH)

    print(f"Setting executable permissions for {PROVER9_TARGET_PATH}")
    os.chmod(PROVER9_TARGET_PATH, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)

    #print(f"Removing Prover9 clone {PROVER9_CLONE_PATH}")
    #shutil.rmtree(PROVER9_CLONE_PATH)

    print("Prover9 fully setup.")

if args.include_libraries:
    print("Installing scipy and numpy.")

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "scipy"])

    print("Setup complete.")


