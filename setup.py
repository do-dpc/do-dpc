from setuptools import setup, find_packages


def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return f.read().splitlines()


setup(
    name="do-dpc",
    version="1.0.0",
    author="Sebastian Graf",
    author_email="sebastian.graf@gmx.ch",
    description="Framework software package for the Data-Driven Predictive Control (DPC) algorithm with visual examples.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/do-dpc/do-dpc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[
        "pylint~=3.3.3",
        "pytest~=8.3.4",
        "pytest-cov~=6.0.0",
        "numpy~=1.26.4",
        "scipy~=1.15.1",
        "black~=25.1.0",
        "mypy~=1.15.0",
        "pre-commit~=4.1.0",
        "cvxpy~=1.6.0",
        "onnx>=1.13.0",
        "ipykernel~=6.29.5",
        "packaging~=24.2",
        "asyncua~=1.1.5",
        "control~=0.10.1",
        "gymnasium~=1.1.1",
        "pygame~=2.6.1",
        "box2d-py~=2.3.8",
        "moviepy~=2.1.2",
        "matplotlib~=3.10.0",
        "ipython~=8.32.0",
        "PyVirtualDisplay~=3.0",
        "tqdm~=4.67.1",
    ],
)
