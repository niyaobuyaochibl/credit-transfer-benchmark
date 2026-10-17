"""Tools for evaluating how models of human choice transfer when the feedback format changes."""
from .observation import common_observation, project
from .metrics import person_scores, calibration, per_session
from .contrast import select_baseline, stratified_bootstrap, transfer_contrast
from .gate import Gate, sha256
from .identifiability import recoverability

__version__ = '0.1.0'
