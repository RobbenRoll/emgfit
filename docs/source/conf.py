# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import emgfit
import sys
sys.path.insert(0, os.path.abspath('../emgfit'))


# -- Project information -----------------------------------------------------

project = 'emgfit'
copyright = '2020, Stefan Paul'
author = 'Stefan Paul'

# Automatically grab version info for the project, acts as replacement for
# |version| and |release|, also used in various other places
version = release = emgfit.__version__.split('+', 1)[0]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    #'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
] # enable napoleon ('sphinxcontrib.napoleon')

# Add mappings to other package docs
intersphinx_mapping = {
    'py': ('http://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'lmfit': ('https://lmfit.github.io/lmfit-py', None)
}

#numpydoc_show_class_members = True

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

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
html_theme = 'sphinx_rtd_theme'

# Temporary work-around for spacing problem between parameter and parameter
# type in the doc, see https://github.com/numpy/numpydoc/issues/215. The bug
# has been fixed in sphinx (https://github.com/sphinx-doc/sphinx/pull/5976) but
# through a change in sphinx basic.css except rtd_theme does not use basic.css.
# In an ideal world, this would get fixed in this PR:
# https://github.com/readthedocs/sphinx_rtd_theme/pull/747/files
#def setup(app):
#    app.add_stylesheet("basic.css")

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Fixed missing separation between parameter names and types by adding:
#.classifier:before {
#    font-style: normal;
#    margin: 0.5em;
#    content: ":";
#}
#to a _static/custom.css file and activated it here by adding:
html_css_files = ['custom.css']
