from setuptools import setup

setup(
    name="nestedlogit",
    version="0.1",
    description="TODO",
    url="https://github.com/lnediak/nestedlogit",
    author="lnediak",
    license="EPL",
    packages=["nestedlogit"],
    python_requires=">=3.7",
    # TODO: appropriate version numbers
    install_requires=[
        "casadi",
        "cyipopt",
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
    ],
)
