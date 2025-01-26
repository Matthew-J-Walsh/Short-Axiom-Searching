import os
import stat
import subprocess
import urllib.request
from setuptools import setup
from setuptools.command.install import install


VAMPIRE_BINARY_URL = "https://github.com/vprover/vampire/releases/download/v4.9casc2024/vampire"
VAMPIRE_TARGET_PATH = "theorem_provers/vampire"
PROVER9_REPO_URL = "https://github.com/ai4reason/Prover9"
PROVER9_CLONE_PATH = "build/Prover9"
PROVER9_TARGET_PATH = "my_library/bin/prover9"


class SASInstallCommand(install):
    def run(self):
        install.run(self)

        vampire_dir = os.path.dirname(VAMPIRE_TARGET_PATH)
        if not os.path.exists(vampire_dir):
            os.makedirs(vampire_dir)

        print(f"Downloading vampire binary from {VAMPIRE_BINARY_URL}")
        urllib.request.urlretrieve(VAMPIRE_BINARY_URL, VAMPIRE_TARGET_PATH)

        print(f"Setting executable permissions for {VAMPIRE_TARGET_PATH}")
        os.chmod(VAMPIRE_TARGET_PATH, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)

        print("Vampire fully setup.")

        print(f"Cloning Prover9 to {PROVER9_CLONE_PATH}")
        if not os.path.exists(PROVER9_CLONE_PATH):
            subprocess.check_call(["git", "clone", PROVER9_REPO_URL, PROVER9_CLONE_PATH])

        print("Building Prover9")
        subprocess.check_call(["make", "all"], cwd=PROVER9_CLONE_PATH)

        compiled_prover9_path = os.path.join(PROVER9_CLONE_PATH, "bin", "prover9")
        assert os.path.exists(compiled_prover9_path), "Prover9 build failure"
        print(f"Moving Prover9 binary to {PROVER9_TARGET_PATH}")

        print(f"Setting executable permissions for {PROVER9_TARGET_PATH}")
        os.chmod(PROVER9_TARGET_PATH, stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)

        print("Prover9 fully setup.")

setup(
    name="ShortAxiomSearching",
    version="0.0.0",
    packages=["ShortAxiomSearching"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy"
    ],
    cmdclass={
        'install': SASInstallCommand
    },
    package_data={
        "ShortAxiomSearching": ["theorem_provers/vampire", "theorem_provers/prover9"]
    },
    description="Methods for discovering short axioms in finite modeled logics.",
    author="Matthew Walsh",
    author_email="matthewwalsh2021@u.northwestern.edu",
    url="https://github.com/Matthew-J-Walsh/Short-Axiom-Searching",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.11.5",
)

