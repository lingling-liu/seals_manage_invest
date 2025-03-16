from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from setuptools import find_packages
from setuptools import setup

extensions = [
        Extension(
          "hazelbean.calculation_core.cython_functions",
          ["hazelbean/calculation_core/cython_functions.pyx"]),
        Extension(
          "hazelbean.calculation_core.aspect_ratio_array_functions",
          ["hazelbean/calculation_core/aspect_ratio_array_functions.pyx"])
]
setup(
    name='hazelbean',
    packages=find_packages(),
    include_package_data=True,
    version='1.5.4',
    description='Geospatial research tools',
    author='Justin Andrew Johnson',
    url='https://github.com/jandrewjohnson/hazelbean',
    # download_url='https://github.com/jandrewjohnson/hazelbean/releases/hazelbean_x64_py3.6.3/dist/hazelbean-0.3.0_x64_py3.6.3.tar.gz',
    keywords=['geospatial', 'raster', 'shapefile'],
    classifiers=[],
    install_requires=[
    ],
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions)
)
