"""
pyEEG package for analyzing EEG with speech and word-level features.

 import pyeeg.* and have fun decoding!

 2019, Hugo Weissbart
"""

from __future__ import division, print_function, absolute_import
from pkg_resources import get_distribution
from . import cca, io, mcca, models, preprocess, utils, vizu

# Suppressing annoying logging in IPython 7.15.0 (issues with parso?)
# More details: https://github.com/ipython/ipython/issues/10946
import logging
logging.getLogger('parso').setLevel(logging.WARNING)

__version__ = get_distribution('pyeeg').version