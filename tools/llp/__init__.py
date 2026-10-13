"""
# __init__.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Long-lived-particle (LLP) reach tools.

Setting- and model-agnostic tools for decay-in-volume LLP studies, driven by
a Pythia forward flux. The pipeline is four stages, each a BaseTool:

  0. ProductionSpectrumTool (optional): a hosted SM parent + an LLP mass ->
     the normalised rest-frame energy spectrum f_h(x) AND the reduced
     branching fraction B_hat, computed from the amplitude rather than
     supplied. Optional because stage 2 still consumes a DECLARED spectrum:
     this tool is one producer of that data product, not a replacement for
     the seam. Supply your own table and the chain is unchanged.
  1. HarvestForwardFluxTool: a Pythia event sample (evtjsonl-1.0) ->
     weighted forward parent-flux files (per-inelastic-collision weights).
  2. MesonDecayToLLPTool: a parent flux + a DECLARED parent-rest-frame LLP
     energy spectrum (a pinned data product -- a table or a two-body delta,
     not a hard-coded amplitude) -> g^2-stripped LLP records carrying a
     decay-in-flight production vertex.
  3. DecayInVolumeVsCouplingTool / DecayInVolumeVsLifetimeTool: LLP records +
     geometry -> signal yields over a coupling grid (portal g^2 reweighting,
     the usual reach scan) or an explicit lab-frame ctau grid (model-agnostic),
     with exact reweighting, the z_shield absorber, off-axis geometry, and a
     selectable two_track / photon / none acceptance.

The experimental setting (collider-forward vs beam-dump) lives in the run
card, geometry YAML, and cross-section normalization. The production physics
lives in the spectrum DATA PRODUCT, which stages 1-3 never look inside;
ProductionSpectrumTool is an optional, separately validated producer of that
product for hosted SM parents, and adding an interaction there leaves the rest
of the chain untouched.
"""

from .harvest_forward_flux import HarvestForwardFluxTool
from .production_spectrum import ProductionSpectrumTool
from .meson_decay_to_llp import MesonDecayToLLPTool
from .decay_in_volume import (
    DecayInVolumeVsCouplingTool,
    DecayInVolumeVsLifetimeTool,
)

__all__ = [
    "ProductionSpectrumTool",
    "HarvestForwardFluxTool",
    "MesonDecayToLLPTool",
    "DecayInVolumeVsCouplingTool",
    "DecayInVolumeVsLifetimeTool",
]
