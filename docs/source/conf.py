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

# Single backticks render as inline code
default_role = "code"

# =============================================================================
# Extensions
# =============================================================================
extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",  # Extract docstrings from source code
    "sphinx.ext.autosummary",  # Generate summary tables for modules/classes
    "sphinx.ext.intersphinx",  # Link to external project docs (numpy, scipy)
    "sphinx.ext.mathjax",  # Render LaTeX math in docs
    "sphinx.ext.viewcode",  # Add [source] links to API docs
    # Third-party extensions
    "autodocsumm",  # Short method names in class summary tables
    "numpydoc",  # Parse NumPy-style docstrings
    "myst_parser",  # Markdown support (.md files)
    "sphinx_design",  # Cards, grids, tabs for landing pages
    "sphinx_gallery.gen_gallery",  # Generate example gallery from scripts
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

# =============================================================================
# HTML Theme (pydata-sphinx-theme)
# =============================================================================
html_theme = "pydata_sphinx_theme"
html_js_files = [
    "pypi-icon.js",
]

html_theme_options = {
    # Site-wide announcement banner
    "announcement": (
        "⚠️ <strong>Early Development</strong> — latents is not yet stable. "
        "The API is subject to frequent change."
    ),
    # Navigation bar (version switching handled by RTD flyout)
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
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
    "navigation_depth": 3,
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

# =============================================================================
# Autodoc settings
# =============================================================================
# Show type hints in function signatures
autodoc_typehints = "signature"
# Use short names without module prefix
add_module_names = False
# Wrap long signatures across multiple lines (one parameter per line)
maximum_signature_line_length = 80
autodoc_member_order = "bysource"
# Exclude undocumented members (prevents empty attribute blocks)
autodoc_default_options = {
    "members": True,
    "undoc-members": False,  # Don't show members without docstrings
}

# =============================================================================
# Numpydoc settings
# =============================================================================
# NOTE: show_class_members=False suppresses auto-generated Attributes section
# and individual attribute blocks. Properties with docstrings still appear.
# For dataclasses, document fields in Parameters section instead.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
# Prevent common words from rendering as type badges
numpydoc_xref_ignore = {"default", "of", "optional", "or", "shape"}
