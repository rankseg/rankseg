import importlib
import inspect
import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rankseg import __version__ as rankseg_version  # noqa: E402

# -- Project information -----------------------------------------------------

project = "RankSEG"
copyright = "2025, Ben Dai and Zixun Wang"
author = "Ben Dai, Zixun Wang"
release = rankseg_version

# -- General configuration ---------------------------------------------------

master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "sphinx_design",
]

# -- Plausible support
ENABLE_PLAUSIBLE = os.environ.get("READTHEDOCS_VERSION_TYPE", "") in ["branch", "tag"]
html_context = {"enable_plausible": ENABLE_PLAUSIBLE}

# -- autoapi configuration ---------------------------------------------------
autodoc_typehints = "signature"
autoapi_type = "python"
autoapi_dirs = [str(REPO_ROOT / "rankseg")]
autoapi_template_dir = "_templates/autoapi"
autoapi_root = "autoapi/rankseg"
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = False

# -- bibtex configuration -----------------------------------------------------
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"


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
    "css/rankseg_tabs.css",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
templates_path = ["_templates"]

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]


# -- linkcode configuration --------------------------------------------------
def linkcode_resolve(domain, info):
    """Return the GitHub source URL for a documented Python object."""
    if domain != "py" or not info.get("module"):
        return None

    try:
        obj = importlib.import_module(info["module"])
        for part in info.get("fullname", "").split("."):
            if part:
                obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        source_file = inspect.getsourcefile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
        relative_path = Path(source_file).resolve().relative_to(REPO_ROOT)
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return None

    end_line = start_line + len(source_lines) - 1
    git_ref = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "main")
    return f"https://github.com/rankseg/rankseg/blob/{git_ref}/{relative_path.as_posix()}#L{start_line}-L{end_line}"


def autoapi_skip_members(app, what, name, obj, skip, options):
    if what == "attribute":
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", autoapi_skip_members)
