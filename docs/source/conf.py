from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version


project = "pyHECTR"
author = "Radik Batraev"
copyright = "2026, Radik Batraev"

try:
    release = package_version("pyhectr")
except PackageNotFoundError:
    release = "0.1.0"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

autosummary_generate = True

# NumPy-style function docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Show type hints inside parameter descriptions
autodoc_typehints = "description"

# Do not execute large experimental notebooks during documentation builds
nb_execution_mode = "off"

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

templates_path = []
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]

html_theme = "pydata_sphinx_theme"
html_static_path = []

html_theme_options = {
    "github_url": "https://github.com/batradiik/pyHECTR",
    "show_toc_level": 2,
}