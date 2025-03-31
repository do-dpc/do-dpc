Installation
============

Library
-------

This library utilizes the `Mosek Solver <https://www.mosek.com/>`_. While it is possible to use any solver compatible with
``CVXPY``, it is recommended to use Mosek for optimal performance. Please follow the installation instructions on
the `Mosek website <https://www.mosek.com/>`_ to set it up.

To ensure a clean and isolated development environment, it is recommended to use Python's virtual environment (`venv`).

Linux / macOS
^^^^^^^^^^^^^


.. code-block:: sh

    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Windows
^^^^^^^


.. code-block:: sh

    py -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt

Documentation
-------------


Navigate to the ``docs`` folder:

.. code-block:: shell

   cd docs

Create and activate a virtual environment:

.. code-block:: shell

   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

Generate the documentation:

.. code-block:: shell

   make html

Once the HTML files are created, you can serve them locally:

.. code-block:: shell

   python -m http.server --directory build/html 8000

Open your browser and visit:

`http://localhost:8000 <http://localhost:8000>`_
