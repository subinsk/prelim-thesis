"""
MTech Thesis: Explainable AI for Docker Container Malware Detection

This package contains the core implementation of image-based container malware
detection using CNNs and Vision Transformers, with explainability methods.

Author: MTech Student, IIIT Guwahati
Supervisor: Dr. Ferdous Ahmed Barbhuiya
"""

__version__ = "0.1.0"
__author__ = "MTech Student - IIIT Guwahati"

# Package-level imports for convenience
from . import data
from . import models
from . import xai
from . import hilbert
from . import utils
