from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version


project = "pyHECTR"
author = "Radik Batraev"
copyright = "2026, Radik Batraev"

try:
    release = package_version("pyhectr")
except PackageNotFoundError:
    release = "0.2.2"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
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
    "deflist",
    "dollarmath",
]

myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints",
    "**/.ipynb_checkpoints/*",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

html_theme = "sphinx_rtd_theme"
html_title = f"pyHECTR {release}"
html_static_path = ["_static"]

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/pyhectr_logo.svg"