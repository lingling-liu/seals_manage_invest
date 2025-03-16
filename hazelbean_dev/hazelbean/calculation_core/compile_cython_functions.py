from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy



# ext_modules = [Extension('gtap_invest',
#                          ['seals_cython_functions.pyx'],
#                          )]

ext_modules = [Extension('aspect_ratio_array_functions',
                         ['aspect_ratio_array_functions.pyx'],
                         )]

returned = setup(
    name='aspect_ratio_array_functions',
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
