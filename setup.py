from distutils.core import setup
from distutils.extension import Extension
from setuptools import setup, find_packages
import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os

packages=find_packages()
include_package_data=True

extensions = [
        Extension(
        "seals.seals_cython_functions",  # This corresponds to the Python import path
        ["seals/seals_cython_functions.pyx"],  # Path to the .pyx file
    )
]

setup(
    name = 'seals',
    packages = packages,
    version = '1.5.4',
    description = 'Land-use change model and downscaler',
    author = 'Justin Andrew Johnson',
    url = 'https://github.com/jandrewjohnson/seals_dev',
    # download_url = 'https://github.com/jandrewjohnson/hazelbean/releases/hazelbean_x64_py3.6.3/dist/hazelbean-0.3.0_x64_py3.6.3.tar.gz',
    keywords = ['geospatial', 'raster', 'shapefile'],
    classifiers = [],
    ext_modules=cythonize(extensions),
    # ext_modules=[Extension("seals_cython_functions", ["seals/seals_cython_functions.c"]),],
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext},
)
