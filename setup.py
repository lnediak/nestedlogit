from distutils.core import setup, Extension

def main():
    setup(name = "logitthing",
          py_modules = ["logitthing"],
          ext_modules = [Extension("_C_logitthing",
                                   ["C_logitthing.cpp"],
                                   include_dirs = ["cppad_include", "ipopt_include"],
                                   libraries=["ipopt"],
                                   extra_compile_args=["-O0"])])

if __name__ == "__main__":
    main()

