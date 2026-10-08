import os
import platform

import numpy
from setuptools import Extension, setup

IS_WINDOWS = platform.system() == "Windows"

libdir = "lib"
tskroot = os.path.join("..", "git-submodules", "tskit", "c")
tskdir = os.path.join(tskroot, "tskit")
kasdir = os.path.join(tskroot, "subprojects", "kastore")
# libdir must come first so that our copy of haplotype_matching.h is used.
includes = [libdir, "lwt_interface", tskroot, tskdir, kasdir]

hmm_source_files = ["haplotype_matching.c"]
# All of tskit apart from its own copy of haplotype_matching.c.
tsk_source_files = [
    "core.c",
    "tables.c",
    "trees.c",
    "genotypes.c",
    "stats.c",
    "convert.c",
]
kas_source_files = ["kastore.c"]

sources = (
    ["_tskit_hmm_module.c"]
    + [os.path.join(libdir, f) for f in hmm_source_files]
    + [os.path.join(tskdir, f) for f in tsk_source_files]
    + [os.path.join(kasdir, f) for f in kas_source_files]
)

libraries = ["Advapi32"] if IS_WINDOWS else []

extra_compile_args = [] if IS_WINDOWS else ["-std=c99"]

_tskit_hmm_module = Extension(
    "_tskit_hmm",
    sources=sources,
    extra_compile_args=extra_compile_args,
    libraries=libraries,
    undef_macros=["NDEBUG"],
    include_dirs=includes + [numpy.get_include()],
)

setup(
    ext_modules=[_tskit_hmm_module],
)
