"""
# test_docstring_contract.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

The tool description is the agent's only channel, and length is a cost.

WHAT THE TRIALS SHOWED. `ComputeDecayRate`'s description reached 9,238
characters, and there is direct evidence it is not being absorbed: the `2*g`
normalisation was stated plainly and ignored, and three separate trials
reported that there were "no examples with spin-1/2 particles" when two of the
worked examples use `"spin": 0.5`. Meanwhile the `spin` field -- which decides
WHICH PROCESS is computed, and whose misuse caused every catastrophic trial we
have seen -- got nine words inside a paragraph about mass symbols, and the
convention (fermion 0.5, vector 1, scalar 0) was never stated anywhere.

So these tests pin an allocation of attention, not a word count: the spin
convention must be stated explicitly and early, the prefactor material must
stay short enough not to crowd it, and ComputeCrossSection must document its
inputs and outputs at all -- it had no Input or Output section, no worked
example, and no mention of the m<label> default, so the xsec tasks faced the
mass trap undocumented.
"""

import re

import pytest

from tools.eda.compute_rate_tools import ComputeDecayRate, ComputeCrossSection

DECAY = ComputeDecayRate.__doc__ or ""
XSEC = ComputeCrossSection.__doc__ or ""


# ---------------------------------------------------------------------------
# The spin convention: stated, and stated early.
# ---------------------------------------------------------------------------

def test_the_spin_convention_is_stated_explicitly():
    """Not implied by an example. The three values, named."""
    head = DECAY
    assert re.search(r"0\.5.{0,40}fermion|fermion.{0,40}0\.5", head, re.S | re.I), \
        "no statement that a fermion is spin 0.5"
    assert re.search(r"vector.{0,30}\b1\b|\b1\b.{0,30}vector", head, re.S | re.I)
    assert re.search(r"scalar.{0,30}\b0\b|\b0\b.{0,30}scalar", head, re.S | re.I)


def test_the_spin_convention_comes_before_the_prefactor_material():
    """Attention is ordered. The field that decides which process is computed
    should not sit below a discussion of factors of two."""
    i_spin = DECAY.lower().find("spin 0.5")
    if i_spin < 0:
        i_spin = DECAY.lower().find("fermion is 0.5")
    i_pref = DECAY.find("IDENTICAL PARTICLES")
    assert i_spin >= 0, "spin convention not found at all"
    assert i_pref < 0 or i_spin < i_pref, "spin convention buried below prefactors"


def test_the_spin_field_is_flagged_as_deciding_the_process():
    assert re.search(r"spin.{0,120}(decide|determine|select)", DECAY, re.S | re.I), \
        "nothing says the spins choose the vertex/process"


# ---------------------------------------------------------------------------
# Length. Not vanity: a description that is skipped is not documentation.
# ---------------------------------------------------------------------------

def test_the_decay_description_is_not_longer_than_it_was():
    assert len(DECAY) <= 9238, "description grew; it was already not being read"


def test_the_decay_description_is_materially_shorter():
    assert len(DECAY) < 7500, f"still {len(DECAY)} chars"


# ---------------------------------------------------------------------------
# ComputeCrossSection parity: it documented neither its inputs nor outputs.
# ---------------------------------------------------------------------------

def test_the_cross_section_documents_its_inputs():
    assert re.search(r"^\s*Input", XSEC, re.M), "no Input section"


def test_the_cross_section_documents_its_outputs():
    assert re.search(r"^\s*Output", XSEC, re.M), "no Output section"


def test_the_cross_section_states_the_mass_symbol_convention():
    """The trap 2.16.1 fixed for decays was undocumented here."""
    assert "m<label>" in XSEC


def test_the_cross_section_states_the_spin_convention():
    assert re.search(r"0\.5", XSEC), "no spin convention"


def test_the_cross_section_has_a_worked_example():
    assert '"topology"' in XSEC, "no example diagram"


def test_both_tools_keep_the_coupling_convention():
    for doc in (DECAY, XSEC):
        assert "2*g" in doc
