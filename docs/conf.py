import os
import sys

import requests


sys.path.insert(0, os.path.abspath('..'))
print(sys.path)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xai-compare'
copyright = '2024, Ran Emuna'
author = 'Ran Emuna'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# # The short X.Y version.
# version = "latest"
# # The full version, including alpha/beta/rc tags.
# release = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "numpydoc",
    "nbsphinx",  # Allows parsing Jupyter notebooks
    "myst_parser",  # Allows parsing Markdown, such as CONTRIBUTING.md
    "sphinx_github_changelog",
]

autodoc_default_options = {"members": True, "inherited-members": True}

# # Retrieve the GitHub token from the configuration file
# config = configparser.ConfigParser()
# config.read('../../xai_compare/config.ini')
#
# sphinx_github_changelog_token = config.get('github', 'token')

# Retrieve the GitHub token from the environment variable
sphinx_github_changelog_token = os.getenv('GITHUB_TOKEN')

# Make sure the autogenerated targets are unique
autosectionlabel_prefix_document = True

autosummary_generate = True
numpydoc_show_class_members = False

# Do not create toctree entries for each class/function
toc_object_entries = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Release notes configuration ------------------------------------------

# Make available a URL that points to the latest unreleased changes


def get_latest_tag() -> str:
    """Query GitHub API to get the most recent git tag"""
    url = "https://api.github.com/repos/emunaran/xai-compare/releases/latest"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["tag_name"]


_latest_tag = get_latest_tag()
_url = f"https://github.com/emunaran/xai-compare/compare/{_latest_tag}...main"

# Make an RST substitution that inserts the correct hyperlink
rst_epilog = f"""
.. |unreleasedchanges| replace:: {_latest_tag}...master
.. _unreleasedchanges: {_url}
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    #'canonical_url': '',
    "logo_only": False,
    # "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#343131",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "../docs/images/xai-compare_logo.png"

html_static_path = ['_static']

