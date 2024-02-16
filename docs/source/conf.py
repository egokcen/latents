"""Configuration file for the Sphinx documentation builder."""

# Build locally using sphinx autobuild:
#     $ sphinx-autobuild docs/source/ docs/build/html/

# Path setup
import os
import sys
from datetime import datetime
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath("../.."))

# Project information
project = "latents"
author = "Evren Gokcen"
copyright = f"{datetime.now().year}, {author}"
release = get_version("latents")
version = ".".join(release.split(".")[:2])  # Take only major/minor

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "myst_parser",
    "nbsphinx",
]
myst_enable_extensions = ["colon_fence"]  # For using MyST Parser with sphinx design
exclude_patterns = []

# HTML theme options
html_theme = "pydata_sphinx_theme"  # Use the PyData theme
html_js_files = [
    "pypi-icon.js",  # PyPI icon
]

html_theme_options = {
    # Navigation bar items
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 4,  # Number of items before a "More" dropdown
    # Project logo
    "logo": {
        "text": "latents",
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/egokcen/latents",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/latents",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
    ],
    # Footer items and style
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    # Primary navigation bar
    "show_nav_level": 0,  # Collapse navigation to the top-level items
    "navigation_depth": 2,  # Control how many levels of navigation are shown
    # Set up the sidebar, on all pages but the index page
    "secondary_sidebar_items": {
        "**/*": ["page-toc"],
    },
}
html_context = {
    "github_user": "egokcen",
    "github_repo": "latents",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "dark",
}
html_static_path = ["_static"]
html_css_files = ["latents.css"]

# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
