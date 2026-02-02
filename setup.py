from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

c_args = ["-O3", "-std=c++14"]

if sys.platform == "darwin":
    c_args += ["-stdlib=libc++", "-mmacosx-version-min=10.15"]

ext_modules = [
    Pybind11Extension(
        "fnblur",
        ["fast_blur.cpp"],
        extra_compile_args=c_args,
    ),
]

setup(
    name="fnblur",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
