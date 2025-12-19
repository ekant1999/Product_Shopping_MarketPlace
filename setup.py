"""Setup script for mixed_precision package and CUDA extension."""
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys


class CMakeBuild(build_ext):
    """Build the mp_kernels extension with CMake. Optional on systems without CUDA."""

    def run(self):
        build_dir = os.path.join(self.build_temp, "build")
        os.makedirs(build_dir, exist_ok=True)
        source_dir = os.path.dirname(os.path.abspath(__file__))
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath("mp_kernels")))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        try:
            subprocess.check_call(["cmake", source_dir, *cmake_args], cwd=build_dir)
            subprocess.check_call(["cmake", "--build", ".", "-j"], cwd=build_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print("Warning: Could not build mp_kernels (CUDA extension).", e)
            print("FP32 baseline and JAX fallbacks will work; mixed-precision kernels disabled.")

    def build_extension(self, ext):
        if ext.name == "mp_kernels":
            self.run()
        else:
            super().build_extension(ext)


setup(
    name="mixed_precision",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax[cuda12]>=0.4.20",
        "jaxlib>=0.4.20",
        "flax>=0.8.0",
        "optax>=0.1.7",
        "tensorflow-datasets>=4.9.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    ext_modules=[__import__("setuptools").Extension("mp_kernels", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    extras_require={"dev": ["pytest"]},
)
