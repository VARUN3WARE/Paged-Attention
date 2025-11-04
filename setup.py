from setuptools import setup, find_packages

setup(
    name="paged_attention",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pytest>=7.4.0",
    ],
    python_requires=">=3.10",
)