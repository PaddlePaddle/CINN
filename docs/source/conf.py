# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join(cur_path, '../../python'))

# -- Project information -----------------------------------------------------

project = 'cinn'
copyright = '2020, cinn team'
author = 'cinn Team'

# The full version, including alpha/beta/rc tags
release = '0.1-alpha'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'recommonmark',
    'breathe',
    'exhale',
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

examples_dirs = ["../../tutorials"]
gallery_dirs = ["tutorials"]

from sphinx_gallery.sorting import ExplicitOrder

subsection_order = ExplicitOrder([
    "../../tutorials",
])

sphinx_gallery_conf = {
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('cinn', 'numpy'),
    'reference_url': {
        'tvm': None,
        'matplotlib': 'http://matplotlib.org',
        'numpy': 'http://docs.scipy.org/doc/numpy-1.9.1'
    },
    'examples_dirs': examples_dirs,
    'gallery_dirs': gallery_dirs,
    'subsection_order': subsection_order,
    'filename_pattern': '/tutorials',
    'find_mayavi_figures': False,
    'expected_failing_examples': []
}

###################################################################
# Setup the breathe extension

breathe_projects = {"CINN Project": "./doxygen_output/xml"}
breathe_default_project = "CINN Project"

# Setup the exhale extension
exhale_args = {
    # These arguments are required
    "containmentFolder": "./cpp",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ Symbols",
    "doxygenStripFromPath": "..",
    # Suggested optional arguments
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT = ../../cinn/frontend ../../cinn/lang"
}

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'

# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'
