# Do DPC

![Pipeline Status](https://gitlab.ethz.ch/do-ddpc/do-ddpc/badges/main/pipeline.svg)
![Coverage](https://gitlab.ethz.ch/do-ddpc/do-ddpc/badges/main/coverage.svg)
[![Documentation Status](https://readthedocs.org/projects/do-ddpc/badge/?version=latest)](https://do-ddpc.readthedocs.io/en/latest/?badge=latest)

The purpose of this code is to provide a Framework for Data-Driven Predictive Control (DPC) with illustrative examples.

Please visit the extensive [documentation](https://do-ddpc.readthedocs.io/en/latest/), kindly hosted on `readthedocs`.

## Citing do-ddpc

Please follow the [instructions](https://do-ddpc.readthedocs.io/en/latest/getting_started/credit.html)
if you want to use **do-ddpc** for published work.

## Getting Started

This library utilizes the [Mosek Solver](https://www.mosek.com/). While it is possible to use any solver compatible with
`CVXPY`, it is recommended to use Mosek for optimal performance. Please follow the installation instructions on
the [Mosek website](https://www.mosek.com/) to set it up.

To ensure a clean and isolated development environment, it is recommended to use Python's virtual environment (`venv`).

Linux / macOS

```sh
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows

```sh
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Building the documentation

Navigate to the `docs` Folder:

```shell
cd docs
```

Create and activate a virtual environment:

```sh
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Generate the documentation:

```shell
make html
```

Once the HTML files are created, you can serve them locally:

```shell
python -m http.server --directory build/html 8000
```

Open your browser and visit:

http://localhost:8000


## Folder structure

The code contains the following files and folders:

```
do-ddpc/
├── do_ddpc/                      # Core Python source code for the DPC library
│   ├── control_utils/            # Utilities for control systems (PID, noise generators, etc.)
│   ├── ddpc/                     # Core DPC algorithms and implementations
│   │   ├── ddpc.py               # Main DPC implementation
│   │   ├── ddpc_structs.py       # Data structures for DPC
│   ├── utils/                    # General utility modules (logging, path management, etc.)
├── docs/                         # Documentation files (Sphinx)
│   ├── build/                    # Build artifacts for generated documentation
│   ├── source/                   # Source files for documentation
│   │   ├── _static/              # Static assets (images, CSS, etc.)
│   │   ├── _templates/           # Templates for documentation structure
│   │   ├── ddpc_methods/         # Documentation for different DPC methods
│   │   ├── getting_started/      # Guides and introductory documentation
│   │   ├── conf.py               # Sphinx configuration file
│   │   ├── index.rst             # Main entry point for documentation
│   ├── .readthedocs.yaml         # Configuration for Read the Docs
│   ├── Makefile                  # Makefile for building the documentation
│   ├── make.bat                  # Windows batch script for building documentation
├── tests/                        # Test suite for the DPC library
│   ├── fixtures/                 # Pytest fixtures for setting up test cases
│   ├── system_tests/             # System-level integration tests
│   ├── unit_tests/               # Unit tests for individual modules
├── .gitlab-ci.yml                # CI/CD pipeline configuration for GitLab
├── .pylintrc                     # Linter configuration for Python code
├── CITATION.cff                  # Citation file for academic references
├── confest.py                    # Pytest configuration file
├── LICENSE.txt                   # Licensing information
├── README.md
├── requirements.txt
└── setup.py                      # Setup script for packaging and installation
```
