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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os.path as osp

import furo
import nbsphinx
import renku_sphinx_theme

# import sphinx.apidoc
# -- Project information -----------------------------------------------------

project = 'RankSEG'
copyright = '2025, Ben Dai and Zixun Wang'
author = 'Ben Dai, Zixun Wang'
# The full version, including alpha/beta/rc tags
# release = '0.10'

import os
import sys

sys.path.append('.')
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../rankseg'))
# sys.path.append('../..')
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(1, os.path.dirname(os.path.abspath("../")) + os.sep + "feature_engine")
# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
master_doc = 'index'
extensions = [
	# 'sphinx.ext.autodoc',
    'autoapi.extension',
    # "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    # 'sphinx.ext.autosummary',
    # 'sphinx_gallery.gen_gallery',
	# 'numpydoc',
	'nbsphinx',
	'sphinx_design',
	]

# -- Plausible support
ENABLE_PLAUSIBLE = os.environ.get("READTHEDOCS_VERSION_TYPE", "") in ["branch", "tag"]
html_context = {"enable_plausible": ENABLE_PLAUSIBLE}

# -- autoapi configuration ---------------------------------------------------
autodoc_typehints = "signature"  # autoapi respects this

autoapi_type = "python"
autoapi_dirs = ['../../rankseg/']
autoapi_template_dir = "_templates/autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = True


# -- custom auto_summary() macro ---------------------------------------------
def contains(seq, item):
    """Jinja2 custom test to check existence in a container.

    Example of use:
    {% set class_methods = methods|selectattr("properties", "contains", "classmethod") %}

    Related doc: https://jinja.palletsprojects.com/en/3.1.x/api/#custom-tests
    """
    return item in seq


def prepare_jinja_env(jinja_env) -> None:
    """Add `contains` custom test to Jinja environment."""
    jinja_env.tests["contains"] = contains


autoapi_prepare_jinja_env = prepare_jinja_env

# Custom role for labels used in auto_summary() tables.
rst_prolog = """
.. role:: summarylabel
"""

# Related custom CSS
html_css_files = [
    "css/label.css",
]

# autosummary_generate = True
# numpydoc_show_class_members = False
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
# autodoc_mock_imports = ['numpy']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme_path = [hachibee_sphinx_theme.get_html_themes_path()]


html_theme = 'furo'

# html_permalinks_icon = '§'

# html_permalinks_icon = 'alpha'
# html_theme = 'sphinxawesome_theme'

# import sphinx_theme_pd
# html_theme = 'sphinx_theme_pd'
# html_theme_path = [sphinx_theme_pd.get_html_theme_path()]

# import solar_theme
# html_theme = 'solar_theme'
# html_theme_path = [solar_theme.theme_path]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_css_files = [
#     'css/custom.css',
# ]

def autoapi_skip_members(app, what, name, obj, skip, options):
    if what == "attribute":
        skip = True
    return skip

def setup(sphinx):
    sphinx.connect("autoapi-skip-member", autoapi_skip_members)

