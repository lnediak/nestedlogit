from distutils.core import setup, Extension
import numpy
import os

def main():
    setup(name = "logitthing",
          py_modules = ["logitthing"],
          ext_modules = [Extension("_C_logitthing",
                                   ["C_logitthing.cpp"],
                                   include_dirs = [numpy.get_include(), "cppad_include", "ipopt_include"],
                                   library_dirs = [os.getcwd(),
                                                   os.path.join(numpy.get_include(), "../../random/lib"),
                                                   os.path.join(numpy.get_include(), "../../core/lib")],
                                   libraries=["ipopt", "npyrandom", "npymath"],
                                   #extra_compile_args=["-O0"]
                                   )])

if __name__ == "__main__":
    main()

