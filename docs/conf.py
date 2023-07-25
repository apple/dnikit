#
# Copyright 2020 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import dnikit

# Project information
project = 'DNIKit'
copyright = '2023 Apple Inc. All rights reserved.'
author = 'Apple, Inc.'
version = dnikit.__version__
release = version

# Required extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

# The suffix(es) of source filenames, multiple as a list of strings:
source_suffix = '.rst'

# Other high level settings
master_doc = 'index'
language = "en"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = None
templates_path = ['_templates']


# HTML Options
html_theme = 'sphinx_book_theme'
html_theme_options = {
    'use_fullscreen_button': False,
    'show_toc_level': 2,
}
html_static_path = ['_static']
html_css_files = ['custom.css']
htmlhelp_basename = 'dnikitdoc' # Output file base name for HTML help builder.

# Latex output config
latex_elements = {}
latex_documents = [
    (master_doc, 'dnikit.tex', 'DNIKit Documentation',
     author, 'manual'),
]

# Manual page output config
man_pages = [
    (master_doc, 'dnikit', 'DNIKit Documentation',
     [author], 1)
]


# Texinfo output config
texinfo_documents = [
    (master_doc, 'dnikit', 'DNIKit Documentation',
     author, 'dnikit', 'Python toolkit for introspection of ML datasets and models.',
     'Miscellaneous'),
]

# Epub output config
epub_title = project
epub_exclude_files = ['search.html']

# Extension config
todo_include_todos = True
autodoc_typehints = "description"
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None)
}

# nbsphinx config
nbsphinx_allow_errors = True
nbsphinx_execute = 'always'
nbsphinx_timeout = 600
nbsphinx_prolog = '''
.. note:: This page was generated from a Jupyter notebook.
          The original can be downloaded from
          `here <https://raw.github.com/apple/dnikit/main/{{env.docname}}.ipynb>`_.
'''
