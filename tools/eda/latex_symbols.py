"""
# latex_symbols.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

HEP symbol vocabulary for TeXForm output.

Mathematica's TeXForm renders a bare multi-letter symbol as \\text{ab}, which
typesets upright rather than as a subscripted variable. tools.wolfram applies
a purely typographic default (\\text{Xy} -> X_y); this module supplies the
physics reading on top of it, for the symbol names the EDA codegen actually
emits.

Most entries below agree with the generic rule and are listed for
explicitness. The ones that do NOT, and are the reason this module exists:

    mfbar   -> m_{\\bar{f}}    (an overbar, not the three letters "bar")
    mf1     -> m_{f_1}        (a subscripted generation index)
    mProp0  -> m_{prop}       (an internal propagator label, not "Prop0")

Pass ``clean_latex_symbols`` as ``latex_postprocess`` to a WolframRunner to
get this reading instead of the generic one.
"""

from tools.wolfram import tidy_text_symbols

# Order matters: longer patterns first, so mfbar is not eaten by mf.
HEP_SYMBOL_MAP = {
    # Masses
    r"\text{mfbar}": r"m_{\bar{f}}",
    r"\text{mf1}": r"m_{f_1}",
    r"\text{mf2}": r"m_{f_2}",
    r"\text{mf}": r"m_f",
    r"\text{mS}": r"m_S",
    r"\text{mV}": r"m_V",
    r"\text{mH}": r"m_H",
    r"\text{mW}": r"m_W",
    r"\text{mZ}": r"m_Z",
    r"\text{mProp0}": r"m_{\text{prop}}",
    # Couplings
    r"\text{gS}": r"g_S",
    r"\text{gP}": r"g_P",
    r"\text{gV}": r"g_V",
    r"\text{gA}": r"g_A",
    r"\text{gL}": r"g_L",
    r"\text{gR}": r"g_R",
    # Yukawas
    r"\text{yb}": r"y_b",
    r"\text{yt}": r"y_t",
    r"\text{ye}": r"y_e",
}


def clean_latex_symbols(tex: str) -> str:
    r"""Rewrite TeXForm output using the HEP symbol vocabulary.

    Applies :data:`HEP_SYMBOL_MAP` first, then falls back to the generic
    typographic rule for any \text{...} the map did not name.
    """
    for pattern, replacement in HEP_SYMBOL_MAP.items():
        tex = tex.replace(pattern, replacement)
    return tidy_text_symbols(tex)
