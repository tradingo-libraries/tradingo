import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                name="tradingo._backtest",
                sources=["src/tradingo/_backtest.pyx"],
            ),
        ],
        gdb_debug=False,
    ),
    include_dirs=[
        np.get_include(),
    ],
)
