# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'numpydoc',  # Enable numpydoc
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'  # Use the pydata-sphinx-theme
html_static_path = ['_static']

# PyData Sphinx Theme Options
html_theme_options = {
    "show_toc_level": 2,
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Suchun-sv/VectorMerge",  # Please replace with your GitHub URL
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "VectorMerge",
        # "image_dark": "_static/logo-dark.svg", # You can add a logo here
    },
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "Beining Yang", # Please replace with your GitHub username
    "github_repo": "VectorMerge",
    "github_version": "main",
    "doc_path": "docs/source",
} 