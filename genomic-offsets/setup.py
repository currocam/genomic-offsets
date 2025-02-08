from setuptools import setup, find_packages

setup(
    name="genomic_offsets",
    author="Curro Campuzano",
    author_email="campuzanocurro@gmail.com",
    license="MIT",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numba>=0.61.0,<0.62",
        "numpy>=2.1.3,<3",
        "scipy>=1.15.1,<2",
        "rpy2>=3.5.11,<4",
        "statsmodels>=0.14.4,<0.15",
        "numdifftools>=0.9.41,<0.10"
    ],
    python_requires=">=3.7",
)