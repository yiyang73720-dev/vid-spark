"""Mechanical checks for Mark Minervini's SEPA method.

Trend (8-point Trend Template + Weinstein Stage 2), Earnings (EPS / revenue
acceleration, margins), and Specific entry point (VCP pivot). The Catalyst
pillar needs judgment and is left to the ``/sepa-screen`` workflow.
"""

from .config import Config
from .screen import evaluate, scan

__all__ = ["Config", "evaluate", "scan"]
