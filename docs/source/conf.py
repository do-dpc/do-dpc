# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.append(os.path.abspath(os.path.join("..", "..", "")))
sys.path.append(os.path.abspath(os.path.join("..", "")))
sys.path.append(os.path.abspath(os.path.join(".")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "do-ddpc"
copyright = "2025, Sebastian Graf"
author = "Sebastian Graf"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

graphviz_output_format = "svg"

autosummary_generate = True

mathjax3_config = {
    "extensions": ["tex2jax.js"],
    "jax": ["input/TeX", "output/HTML-CSS"],
}

templates_path = ["_templates"]
exclude_patterns = []

html_static_path = ["_static"]

html_theme = "sphinx_book_theme"
html_title = "Do DPC"

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://gitlab.ethz.ch/do-ddpc/do-ddpc",
    "use_repository_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "show_navbar_depth": 1,
    "logo": {
        "text": html_title,
    },
}

html_show_sourcelink = True

# -- Options for LaTeX output ---------------------------------------------
latex_engine = "pdflatex"
