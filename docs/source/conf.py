"""Configuration file for the Sphinx documentation builder."""

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
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # Third-party extensions
    "numpydoc",
    "myst_parser",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
]

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",  # ::: directives for sphinx-design
    "dollarmath",  # $inline$ and $$block$$ LaTeX math
    "deflist",  # Definition lists
]

# Autosummary configuration
autosummary_generate = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Sphinx-gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": ["../../examples"],  # Path to example scripts
    "gallery_dirs": ["auto_examples"],  # Path to generated gallery
    "filename_pattern": r"\.py$",  # Include all Python files
    "remove_config_comments": True,
    "plot_gallery": "True",
    "download_all_examples": False,
    "line_numbers": False,
    "within_subsection_order": "FileNameSortKey",
    "matplotlib_animations": True,
}

# HTML theme options
html_theme = "pydata_sphinx_theme"
html_js_files = [
    "pypi-icon.js",
]

html_theme_options = {
    # Navigation bar (version switching handled by RTD flyout)
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 4,
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
    "show_nav_level": 0,
    "navigation_depth": 2,
    # Set up the sidebar
    # Sidebar: page-toc for all pages, plus download links for examples
    "secondary_sidebar_items": ["page-toc", "sg_download_links"],
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
autodoc_typehints = "none"  # numpydoc handles type documentation
autodoc_member_order = "bysource"

# Numpydoc settings
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
