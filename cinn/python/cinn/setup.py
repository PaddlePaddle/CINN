import shutil
import os
from setuptools import setup, Distribution

class BinaryDistribution(Distribution):
    '''binary distribution'''
    def has_ext_modules(foo):
        return True

CINN_BINARY_PATH = "/home/chunwei/project/cinn2/cmake-build-debug"
CINN_SOURCE_PATH = "/home/chunwei/project/cinn2"
INSTALL_PATH = os.path.join(CINN_BINARY_PATH, 'python_install')

#python_cinn_dir = os.path.join(CINN_SOURCE_PATH, 'cinn/python/cinn')
#cinn_core_lib_path = os.path.join(CINN_BINARY_PATH, 'cinn/python/cinn_core.so')
#cinn_core_lib_dir = os.path.join(CINN_BINARY_PATH, 'cinn/python')

#shutil.copy(python_cinn_dir, INSTALL_PATH)
#shutil.copy(cinn_core_lib_path, INSTALL_PATH)

PACKAGE_DATA = {'cinn': ['cinn_core.so', ]}
PACKAGE_DIR = {
    'cinn': '',
    #'cinn.libs': cinn_core_lib_dir,
}


setup(name='cinn',
      version='0.1',
      description="CINN -- Compiler Infrastructure for Neural Networks",
      packages = ['cinn'],
      package_dir=PACKAGE_DIR,
      package_data=PACKAGE_DATA,
      distclass=BinaryDistribution,
      #include_package_data=True,
      zip_safe=False,
      )
