"""
# tex_extract.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

TeX-faithful text extraction from TeX-produced PDFs.

Two facts about how TeX writes PDFs drive this module:

1. Math semantics live in the *font*, not in Unicode. ``CMMI`` is Computer
   Modern Math Italic, so every glyph in it is a variable; ``CMSY``/``CMEX``
   carry operators and extensible delimiters; ``MSBM`` carries blackboard
   bold. The ToUnicode CMap discards all of this, which is why a Unicode
   round-trip cannot distinguish ``\\epsilon`` from ``\\varepsilon``, or the
   variable $p$ from the letter p. Reading the font name per character
   recovers it exactly.

2. Fraction bars and radical vinculums are drawn as *filled rectangles*, not
   glyphs, so they are invisible to the text layer entirely. Display math
   therefore arrives as a stream of disconnected fragments. Reading the page's
   path objects puts the structure back.

Glyph->macro mappings are not hand-written; they are derived by
``build_table.py``, which compiles a probe document under several math-font
packages and reads back what each macro rendered to.

Single runtime dependency: pypdfium2.
"""

import ctypes
import json
import re
import statistics
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

_TABLE_PATH = Path(__file__).with_name("tex_table.json")
_T = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
SYMBOLS, RADICALS = _T["symbols"], _T["radicals"]
DELIMS = _T.get("delims", {})
DELIM_KINDS = _T.get("delim_kinds", [])
MAXSEQ = max((len(k.split("\x00")[1]) for k in SYMBOLS), default=1)

#: delimiter kind -> amsmath matrix environment
MATRIX_ENV = {"(": "pmatrix", "[": "bmatrix", "\\{": "Bmatrix",
              "|": "vmatrix", "\\|": "Vmatrix"}

#: font families whose glyphs are mathematical rather than prose
MATH_FAMILIES = re.compile(
    r"CMMI|CMSY|CMEX|MSAM|MSBM|rsfs|LMMath|NewTXMI|txmia|txsy|txex|StandardSymL",
    re.I,
)

#: accents are emitted as standalone spacing glyphs positioned over their base
ACCENTS = {
    "¯": "bar", "ˆ": "hat", "˜": "tilde", "˙": "dot",
    "¨": "ddot", "ˇ": "check", "˘": "breve",
    "´": "acute", "`": "grave", "^": "hat", "~": "tilde",
}

#: characters that must be escaped to survive as literal TeX
TEX_ESCAPE = {"%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "$": r"\$"}

#: glyphs that carry no macro of their own but have a canonical TeX spelling
LITERAL = {"\u2212": "-", "\u2013": "--", "\u2014": "---", "\u2019": "'", "\u201c": "``", "\u201d": "''"}


# --------------------------------------------------------------------------
# page geometry
# --------------------------------------------------------------------------

def _family(name):
    """``ABCDEF+CMMI10`` -> ``CMMI``; drop subset prefix and design size."""
    return re.sub(r"\d+", "", name.split("+")[-1])


def _read_chars(page):
    """Glyphs with font identity and bounding box, in content-stream order."""
    tp = page.get_textpage()
    out = []
    for i in range(tp.count_chars()):
        c = tp.get_text_range(i, 1)
        if not c:
            continue
        buf = ctypes.create_string_buffer(160)
        flags = ctypes.c_int()
        n = pdfium_c.FPDFText_GetFontInfo(tp, i, buf, 160, ctypes.byref(flags))
        x0, y0, x1, y1 = tp.get_charbox(i)
        # Filter on whether this is a real positioned glyph, NOT on whether the
        # character looks like whitespace. TeX's extensible vertical bar is
        # U+000C (form feed) in CMEX, and str.strip() counts form feed as
        # whitespace -- so a whitespace test silently discards every vmatrix
        # fence. Layout separators carry no font and a degenerate box.
        if n <= 1 or (x1 - x0 <= 0 and y1 - y0 <= 0):
            continue
        # The glyph's baseline origin, NOT the bounding box, is what script
        # level must be measured against: a descender such as p or j sits
        # below the baseline, which would otherwise make its own subscript
        # look higher than its base.
        ox, oy = ctypes.c_double(), ctypes.c_double()
        pdfium_c.FPDFText_GetCharOrigin(tp, i, ctypes.byref(ox), ctypes.byref(oy))
        out.append({
            "c": c,
            "fam": _family(buf.raw[:max(0, n - 1)].decode("utf-8", "replace")),
            "size": pdfium_c.FPDFText_GetFontSize(tp, i),
            "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "base": oy.value,
            "xc": (x0 + x1) / 2.0, "yc": (y0 + y1) / 2.0,
        })
    return out


def _read_rules(page, max_thickness=2.0, min_aspect=2.5):
    """Thin horizontal filled paths: fraction bars, vinculums, \\overline."""
    rules = []
    for obj in page.get_objects():
        if obj.type != 2:                      # FPDF_PAGEOBJ_PATH
            continue
        try:
            x0, y0, x1, y1 = obj.get_bounds()
        except Exception:
            continue
        w, h = x1 - x0, y1 - y0
        if 0 < h <= max_thickness and w > min_aspect * max(h, 0.1):
            rules.append({"x0": x0, "x1": x1, "y": (y0 + y1) / 2.0, "w": w})
    return rules


# --------------------------------------------------------------------------
# accent recomposition
# --------------------------------------------------------------------------

def _recompose_accents(chars):
    """Fold standalone accent glyphs into ``\\bar{x}`` on the base beneath.

    Assignment is resolved first and the sequence rebuilt afterwards, so a base
    is annotated in place rather than re-appended (which would duplicate it
    whenever the accent glyph is not immediately adjacent in stream order).
    """
    assigned = {}       # base index -> accent macro
    is_accent = set()
    for i, ch in enumerate(chars):
        accent = ACCENTS.get(ch["c"])
        if not accent:
            continue
        best, best_d = None, None
        for j in (i - 1, i + 1, i - 2, i + 2):
            if not (0 <= j < len(chars)) or j in assigned or j in is_accent:
                continue
            other = chars[j]
            if ACCENTS.get(other["c"]):
                continue
            d = abs(other["xc"] - ch["xc"])
            if d > 0.9 * max(ch["x1"] - ch["x0"], 1.0) + 0.5 * (other["x1"] - other["x0"]):
                continue
            if abs(other["yc"] - ch["yc"]) > 1.5 * ch["size"]:
                continue
            if best_d is None or d < best_d:
                best, best_d = j, d
        if best is not None:
            assigned[best] = accent
            is_accent.add(i)

    out = []
    for i, ch in enumerate(chars):
        if i in is_accent:
            continue
        if i in assigned:
            ch = dict(ch)
            ch["accent"] = assigned[i]
        out.append(ch)
    return out


# --------------------------------------------------------------------------
# token emission
# --------------------------------------------------------------------------

def _lookup(chars, i):
    """Longest single-font glyph run -> macro. Returns (macro, n_consumed)."""
    fam = chars[i]["fam"]
    for n in range(min(MAXSEQ, len(chars) - i), 0, -1):
        run = chars[i:i + n]
        if any(c["fam"] != fam for c in run):
            continue
        key = fam + "\x00" + "".join(c["c"] for c in run)
        if key in SYMBOLS:
            return SYMBOLS[key], n
        if key in RADICALS:
            return RADICALS[key], n
    return None, 1


def _emit(macro, ch):
    if macro is None:
        info = DELIMS.get(ch["fam"] + "\x00" + ch["c"])
        if info is not None:
            # An extensible delimiter piece that no fence pair claimed. Its
            # glyph code is meaningless outside its font (often a control
            # character), so emit the delimiter it stands for.
            kind = DELIM_KINDS[info["kind"]] if info["kind"] < len(DELIM_KINDS) else "("
            if info["side"] == "right":
                kind = {"(": ")", "[": "]", "\\{": "\\}",
                        "\\langle": "\\rangle"}.get(kind, kind)
            return kind + " "
        if ord(ch["c"][0]) < 32:
            return ""            # never leak raw control codes
        tok = TEX_ESCAPE.get(ch["c"]) or LITERAL.get(ch["c"]) or ch["c"]
    elif len(macro) == 1 and macro.isalpha():
        tok = macro                                   # math-italic letter
    elif macro.endswith("}"):
        tok = "\\" + macro
    else:
        tok = "\\" + macro + " "
    if ch.get("accent"):
        tok = "\\%s{%s}" % (ch["accent"], tok.strip())
    return tok


def _is_math(ch):
    return bool(MATH_FAMILIES.search(ch["fam"])) or bool(ch.get("accent"))


# --------------------------------------------------------------------------
# recursive layout parsing
# --------------------------------------------------------------------------

def _linear(chars, body_size, spaces=True):
    """Left-to-right run, recursing into sub/superscript clusters.

    Scripts are gathered as a *group* after their base rather than tracked as a
    running mode, so a base carrying both a subscript and a superscript
    (``m_{jj}^{2}``) emits them as two complete groups instead of interleaving
    them by x-position.
    """
    chars = sorted(chars, key=lambda c: c["x0"])
    out = []
    i = 0
    prev = None
    while i < len(chars):
        ch = chars[i]
        if prev is not None and spaces:
            gap = ch["x0"] - prev["x1"]
            if gap > 0.22 * body_size:
                out.append(" ")
        macro, n = _lookup(chars, i)
        out.append(_emit(macro, ch))
        base_y = ch["base"]
        prev = chars[i + n - 1]
        i += n

        # collect the contiguous run of script-sized glyphs that follows
        j = i
        while j < len(chars) and chars[j]["size"] < 0.95 * body_size:
            j += 1
        if j > i:
            scripts = chars[i:j]
            # partition by identity: glyph dicts compare equal by value, so
            # `in` would misclassify two identical glyphs on one line
            sup, sub, mid = [], [], []
            for s in scripts:
                d = s["base"] - base_y
                if d > 0.10 * body_size:
                    sup.append(s)
                elif d < -0.04 * body_size:
                    sub.append(s)
                else:
                    mid.append(s)
            inner = max(body_size * 0.7, 1.0)
            if sub:
                out.append("_{" + _linear(sub, inner, spaces=False) + "}")
            if sup:
                out.append("^{" + _linear(sup, inner, spaces=False) + "}")
            if mid:
                out.append(_linear(mid, inner, spaces=False))
            prev = scripts[-1]
            i = j
    return "".join(out)


def _split_by_rule(chars, rule, pad=1.0):
    """Partition chars around a rule into (left, above, below, right)."""
    left, above, below, right = [], [], [], []
    for ch in chars:
        if ch["x1"] <= rule["x0"] + pad:
            left.append(ch)
        elif ch["x0"] >= rule["x1"] - pad:
            right.append(ch)
        elif ch["yc"] > rule["y"]:
            above.append(ch)
        else:
            below.append(ch)
    return left, above, below, right


def _parse_region(chars, rules, body_size):
    """Recursively turn a region into TeX, resolving fractions and radicals."""
    if not chars:
        return ""
    if not rules:
        return _linear(chars, body_size)

    # resolve the outermost structure first: the widest rule in the region
    rule = max(rules, key=lambda r: r["w"])
    others = [r for r in rules if r is not rule]
    left, above, below, right = _split_by_rule(chars, rule)

    def recurse(region_chars):
        """Descend into a sub-region, carrying only the rules it contains."""
        if not region_chars:
            return ""
        rx0 = min(c["x0"] for c in region_chars)
        rx1 = max(c["x1"] for c in region_chars)
        ry0 = min(c["yc"] for c in region_chars)
        ry1 = max(c["yc"] for c in region_chars)
        inner_rules = [r for r in others
                       if r["x0"] >= rx0 - 1 and r["x1"] <= rx1 + 1
                       and ry0 - 2 * body_size <= r["y"] <= ry1 + 2 * body_size]
        return _parse_region(region_chars, inner_rules, body_size)

    # A radical glyph immediately left of the rule makes this a vinculum rather
    # than a fraction: the rule covers its argument instead of separating two.
    radical = None
    for ch in left:
        if ch.get("radical") and abs(ch["x1"] - rule["x0"]) < 2.5 * body_size:
            radical = ch

    if radical is not None:
        left = [c for c in left if c is not radical]
        return recurse(left) + "\\sqrt{" + recurse(above + below) + "}" + recurse(right)
    if not above:
        return recurse(left) + "\\overline{" + recurse(below) + "}" + recurse(right)
    if not below:
        return recurse(left) + "\\underline{" + recurse(above) + "}" + recurse(right)
    return (recurse(left) + "\\frac{" + recurse(above) + "}{" + recurse(below) + "}"
            + recurse(right))


# --------------------------------------------------------------------------
# fences and matrices
# --------------------------------------------------------------------------

def _find_fences(chars, body_size):
    """Group delimiter glyphs into fences.

    A tall fence is either one large glyph or a vertical stack of extensible
    pieces at the same x, so pieces sharing a kind, a side and a column are
    merged into a single fence spanning their combined height.
    """
    pieces = []
    for ch in chars:
        info = DELIMS.get(ch["fam"] + "\x00" + ch["c"])
        if info:
            pieces.append((ch, info))
    fences = []
    used = set()
    for i, (ch, info) in enumerate(pieces):
        if i in used:
            continue
        group = [ch]
        used.add(i)
        for j, (other, oinfo) in enumerate(pieces):
            if j in used or oinfo != info:
                continue
            if abs(other["xc"] - ch["xc"]) < 0.4 * body_size:
                group.append(other)
                used.add(j)
        fences.append({
            "kind": info["kind"], "side": info["side"], "glyphs": group,
            "x0": min(g["x0"] for g in group), "x1": max(g["x1"] for g in group),
            "y0": min(g["y0"] for g in group), "y1": max(g["y1"] for g in group),
        })
    return fences


def _match_fences(fences, body_size):
    """Pair each opener with the nearest closer of the same kind and height."""
    # symmetric fences (| and \|) use one glyph for both sides, so they appear
    # in each list and are paired left-to-right by position
    lefts = sorted((f for f in fences if f["side"] in ("left", "both")),
                   key=lambda f: f["x0"])
    rights = sorted((f for f in fences if f["side"] in ("right", "both")),
                    key=lambda f: f["x0"])
    pairs, taken = [], set()
    for lf in lefts:
        best = None
        for k, rt in enumerate(rights):
            if k in taken or rt["x0"] < lf["x1"] or rt is lf:
                continue
            if rt["kind"] != lf["kind"]:
                continue
            # same vertical extent, within a tolerance of one body height
            if (abs(rt["y0"] - lf["y0"]) < body_size
                    and abs(rt["y1"] - lf["y1"]) < body_size):
                best = k
                break
        if best is not None:
            taken.add(best)
            pairs.append((lf, rights[best]))
    return pairs


def _columns(cells, body_size):
    """Find column boundaries from gaps in the horizontal projection."""
    spans = sorted((c["x0"], c["x1"]) for c in cells)
    bounds, cur_end = [], None
    for x0, x1 in spans:
        if cur_end is not None and x0 - cur_end > 0.6 * body_size:
            bounds.append((cur_end + x0) / 2.0)
        cur_end = max(cur_end or x1, x1)
    return bounds


def _render_matrix(content, kind, body_size, rules):
    """Render fenced content as a matrix environment, or None if it is not a
    grid (a single row of a single cell is just a parenthesised group)."""
    if not content:
        return None
    rows = []
    for ch in sorted(content, key=lambda c: (-c["base"], c["x0"])):
        if rows and abs(rows[-1][0]["base"] - ch["base"]) <= 0.5 * body_size:
            rows[-1].append(ch)
        else:
            rows.append([ch])
    # scripts sit on their own baseline; fold short rows into the nearest row
    merged = []
    for row in rows:
        if merged and all(c["size"] < 0.95 * body_size for c in row):
            merged[-1].extend(row)
        else:
            merged.append(row)
    # a \frac inside a cell puts its numerator and denominator on baselines of
    # their own, which would otherwise read as extra matrix rows; the fraction
    # rule is what says they belong to one entry
    rows = _merge_rows_by_rules(merged, rules, body_size)

    bounds = _columns(content, body_size)
    if len(rows) < 2 and len(bounds) < 1:
        return None

    body = []
    for row in rows:
        cells = [[] for _ in range(len(bounds) + 1)]
        for ch in row:
            idx = sum(1 for b in bounds if ch["xc"] > b)
            cells[idx].append(ch)
        rendered = []
        for cell in cells:
            if not cell:
                rendered.append("")
                continue
            cx0 = min(c["x0"] for c in cell)
            cx1 = max(c["x1"] for c in cell)
            local = [r for r in rules if r["x0"] >= cx0 - 1 and r["x1"] <= cx1 + 1]
            rendered.append(_squash(_parse_region(cell, local, body_size)))
        body.append(" & ".join(rendered).rstrip(" &"))

    env = MATRIX_ENV.get(DELIM_KINDS[kind] if kind < len(DELIM_KINDS) else "(")
    grid = " \\\\ ".join(r for r in body if r.strip())
    if env is None:      # no standard environment (e.g. angle brackets)
        opener = DELIM_KINDS[kind]
        closer = {"\\langle": "\\rangle"}.get(opener, opener)
        return "\\left%s \\begin{array}{%s} %s \\end{array} \\right%s" % (
            opener, "c" * (len(bounds) + 1), grid, closer)
    return "\\begin{%s} %s \\end{%s}" % (env, grid, env)


# --------------------------------------------------------------------------
# block segmentation
# --------------------------------------------------------------------------

def _cluster_rows(chars, body_size):
    """Group glyphs into visual rows by vertical position."""
    rows, current = [], []
    for ch in sorted(chars, key=lambda c: (-c["yc"], c["x0"])):
        if current and abs(current[-1]["yc"] - ch["yc"]) > 0.55 * body_size:
            rows.append(current)
            current = []
        current.append(ch)
    if current:
        rows.append(current)
    return rows


def _merge_rows_by_fences(rows, pairs, body_size):
    """Join the rows a matched fence pair encloses.

    A matrix's rows sit on separate baselines, so vertical clustering splits
    them; the enclosing fence is what says they belong to one expression.
    """
    parent = list(range(len(rows)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for lf, rt in pairs:
        touching = [idx for idx, row in enumerate(rows)
                    if any(lf["x0"] - 1 <= c["xc"] <= rt["x1"] + 1
                           and lf["y0"] - 1 <= c["yc"] <= lf["y1"] + 1 for c in row)]
        for a, b in zip(touching, touching[1:]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for idx in range(len(rows)):
        groups.setdefault(find(idx), []).append(idx)
    return [[c for i in sorted(idxs) for c in rows[i]]
            for _, idxs in sorted(groups.items())]


def _merge_rows_by_rules(rows, rules, body_size):
    """Join rows that a rule structurally spans.

    Vertical clustering already bridges most display math into one row, because
    a numerator and denominator are connected by the glyphs at the intervening
    main baseline. Merging is therefore a narrow repair, and must be strict: a
    running text line that merely happens to pass over a rule's x-span is not a
    numerator. A row qualifies only if the majority of its glyphs sit inside
    the rule's horizontal extent.
    """
    parent = list(range(len(rows)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for rule in rules:
        touching = []
        for idx, row in enumerate(rows):
            inside = [c for c in row if rule["x0"] - 1 <= c["xc"] <= rule["x1"] + 1]
            if not inside:
                continue
            near = any(abs(c["yc"] - rule["y"]) < 1.3 * body_size for c in inside)
            contained = len(inside) >= 0.6 * len(row)
            if near and contained:
                touching.append(idx)
        for a, b in zip(touching, touching[1:]):
            union(a, b)

    groups = {}
    for idx in range(len(rows)):
        groups.setdefault(find(idx), []).append(idx)
    blocks = []
    for _, idxs in sorted(groups.items()):
        blocks.append([c for i in sorted(idxs) for c in rows[i]])
    return blocks


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def page_to_tex(page):
    """Convert one page to TeX-flavoured text."""
    raw = _read_chars(page)
    if not raw:
        return ""
    chars = _recompose_accents(raw)
    sizes = [c["size"] for c in chars]
    body_size = statistics.mode([round(s, 1) for s in sizes]) or 10.0
    rules = _read_rules(page)

    # mark radical glyphs via the derived radical table
    for ch in chars:
        if RADICALS.get(ch["fam"] + "\x00" + ch["c"]):
            ch["radical"] = True

    # Fences must be resolved before row clustering is finalised: a matrix's
    # rows sit on separate baselines and only the enclosing fence says they
    # belong together.
    fences = _find_fences(chars, body_size)
    pairs = _match_fences(fences, body_size)

    rows = _cluster_rows(chars, body_size)
    rows = _merge_rows_by_fences(rows, pairs, body_size)
    blocks = _merge_rows_by_rules(rows, rules, body_size)

    # Each rule belongs to exactly one block: the one whose glyphs under the
    # rule's x-span sit closest to it vertically. Without uniqueness a running
    # text line whose descenders reach the rule's height would claim it too,
    # and get spuriously wrapped in \overline.
    assignment = {i: [] for i in range(len(blocks))}
    for rule in rules:
        best, best_d = None, None
        for idx, block in enumerate(blocks):
            under = [c for c in block if rule["x0"] - 1 <= c["xc"] <= rule["x1"] + 1]
            if not under:
                continue
            d = min(abs(c["yc"] - rule["y"]) for c in under)
            if d > 2.0 * body_size:
                continue
            if best_d is None or d < best_d:
                best, best_d = idx, d
        if best is not None:
            assignment[best].append(rule)

    lines = []
    for idx, block in enumerate(blocks):
        lines.append(_render_block(block, assignment[idx], body_size, pairs))
    return "\n".join(l for l in lines if l.strip())


def _render_block(block, rules, body_size, pairs=()):
    """Render one block, wrapping maximal math runs in ``$``."""
    ids = {id(c) for c in block}
    local_pairs = [(lf, rt) for lf, rt in pairs
                   if all(id(g) in ids for g in lf["glyphs"] + rt["glyphs"])]
    if local_pairs:
        rendered = _render_fenced(block, local_pairs, rules, body_size)
        if rendered is not None:
            return rendered

    if rules:
        # structural math: the whole block is one expression
        return "$" + _squash(_parse_region(block, rules, body_size)) + "$"

    ordered = sorted(block, key=lambda c: c["x0"])
    out, in_math = [], False
    i = 0
    while i < len(ordered):
        ch = ordered[i]
        math_here = _is_math(ch)
        if in_math and not math_here and ch["c"] in "0123456789()[]/=+-,.|<>":
            math_here = True
        # preserve inter-word spacing across a mode switch
        if i > 0 and ch["x0"] - ordered[i - 1]["x1"] > 0.22 * body_size:
            out.append(" ")
        if math_here and not in_math:
            out.append("$")
            in_math = True
        elif in_math and not math_here:
            out.append("$")
            in_math = False
        # a maximal same-mode run is rendered together so scripts nest properly
        j = i
        run = []
        while j < len(ordered):
            nxt = ordered[j]
            m = _is_math(nxt)
            if in_math and not m and nxt["c"] in "0123456789()[]/=+-,.|<>":
                m = True
            if m != in_math:
                break
            run.append(nxt)
            j += 1
        out.append(_linear(run, body_size))
        i = j if j > i else i + 1
    if in_math:
        out.append("$")
    return _squash("".join(out))


def _render_fenced(block, pairs, rules, body_size):
    """Render a block containing matched fences, substituting each fenced grid
    for a matrix environment. Returns None if no pair encloses a real grid."""
    pair = max(pairs, key=lambda p: p[0]["y1"] - p[0]["y0"])
    lf, rt = pair
    fence_ids = {id(g) for g in lf["glyphs"] + rt["glyphs"]}

    inside, before, after = [], [], []
    for ch in block:
        if id(ch) in fence_ids:
            continue
        if ch["x1"] <= lf["x0"] + 1:
            before.append(ch)
        elif ch["x0"] >= rt["x1"] - 1:
            after.append(ch)
        else:
            inside.append(ch)

    inner_rules = [r for r in rules if lf["x1"] - 1 <= r["x0"] and r["x1"] <= rt["x0"] + 1]
    matrix = _render_matrix(inside, lf["kind"], body_size, inner_rules)
    if matrix is None:
        # A tall fence around something that is not a grid — \left( \frac{a}{b}
        # \right), a big operator, a single column of one entry. It must still
        # be rendered as a fence, otherwise the extensible delimiter pieces
        # leak into the output as raw control characters.
        opener = DELIM_KINDS[lf["kind"]] if lf["kind"] < len(DELIM_KINDS) else "("
        closer = {"(": ")", "[": "]", "\\{": "\\}", "\\langle": "\\rangle"}.get(
            opener, opener)
        body = _squash(_parse_region(inside, inner_rules, body_size))
        matrix = "\\left%s %s \\right%s" % (opener, body, closer)

    lead = _squash(_parse_region(before, [r for r in rules if r["x1"] <= lf["x0"]],
                                 body_size)) if before else ""
    tail = _squash(_parse_region(after, [r for r in rules if r["x0"] >= rt["x1"]],
                                 body_size)) if after else ""
    return _squash("$" + " ".join(p for p in (lead, matrix, tail) if p) + "$")


def _squash(s):
    s = re.sub(r"\$\s*\$", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def pdf_to_tex(path, pages=None):
    """Convert a PDF to TeX-flavoured text.

    Args:
        path: path to the PDF.
        pages: iterable of 0-based page indices, or None for all pages.
    """
    doc = pdfium.PdfDocument(str(path))
    indices = range(len(doc)) if pages is None else list(pages)
    return "\n".join(page_to_tex(doc[i]) for i in indices)


if __name__ == "__main__":
    import sys
    sel = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    print(pdf_to_tex(sys.argv[1], sel))
