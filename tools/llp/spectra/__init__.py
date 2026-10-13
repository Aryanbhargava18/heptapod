"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

LLP production spectra: hosted SM parents, model-independent three-body
kinematics, and a registry of interaction vertices.

    parents.py     hosted PDG data, one row per hadron, every number cited
    kinematics.py  phase space + Gauss-Legendre quadrature (model-independent)
    vertices.py    THE MODEL -- squared amplitudes; extend here
"""
from . import kinematics, parents, vertices          # noqa: F401
