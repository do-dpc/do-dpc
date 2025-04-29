from setuptools import setup, find_packages

setup(
    name="do-dpc",
    version="1.0.0",
    author="Sebastian Graf",
    author_email="Sebastian Graf",
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
    python_requires='>=3.12',
    install_requires=load_requirements(),
)
