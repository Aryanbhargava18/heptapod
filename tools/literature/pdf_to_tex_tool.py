"""
# pdf_to_tex_tool.py is a part of the HEPTAPOD package.
# Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Convert a TeX-produced PDF into TeX-flavoured text.
"""

import os
import re
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

SCHEMA_VERSION = "pdftex-1.0"

# tex_extract imports pypdfium2 and loads the glyph table at module scope;
# both are deferred to _run() so toolkit startup stays cheap.


class PDFToTeXTool(BaseTool):
    """
    Extract TeX-flavoured text from a PDF produced by LaTeX.

    Recovers math semantics from the embedded font identity rather than from
    Unicode, so ``\\epsilon`` and ``\\varepsilon`` stay distinct, variables stay
    distinguishable from prose, and ``\\frac``/``\\sqrt`` are reassembled from
    the drawn rules that carry them.

    The extracted text is written to a file under ``base_directory``; the tool
    returns the output path and a structural summary rather than the document
    body, so a long paper does not flood the agent's context.
    """

    # --------------------------- Runtime fields --------------------------- #
    pdf_path: str = RuntimeField(
        description="Relative path to the source PDF, inside base_directory."
    )
    output_path: str = RuntimeField(
        default=None,
        description=(
            "Relative path for the extracted text (e.g. 'papers/2401.12345.tex'). "
            "Defaults to the PDF path with a .tex suffix."
        ),
    )
    pages: str = RuntimeField(
        default=None,
        description=(
            "Optional 1-based page selection, e.g. '1-4' or '1,3,7'. "
            "Omit to extract the whole document."
        ),
    )
    # ---------------------------------------------------------------------- #

    # ---------------------------- State fields ---------------------------- #
    base_directory: str = StateField(default=".", description="Base directory for safe paths")
    # ---------------------------------------------------------------------- #

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PDFToTeX"
        self.description = (
            "Extract TeX-flavoured text from a LaTeX-produced PDF. Recovers Greek "
            "letters, operators, sub/superscripts, accents, fractions, radicals and "
            "matrices as TeX macros by reading the embedded math fonts. Writes the "
            "result to a file and returns the path plus a summary. Best for papers "
            "with no arXiv source available; prefer the LaTeX source when there is "
            "one, since it is authoritative. Output is TeX-flavoured for reading, "
            "not guaranteed to compile: upright math styles (\\mathrm) are "
            "unrecoverable, aligned environments flatten, and a fraction inside a "
            "matrix cell may read as an extra row."
        )

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.exists(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel: str) -> Optional[str]:
        """Ensures that the path is within the allowed base directory."""
        full = os.path.abspath(os.path.join(self.base_directory, rel))
        return full if full.startswith(self.base_directory) else None

    @staticmethod
    def _parse_pages(spec: str, n_pages: int):
        """'1-4,7' -> [0,1,2,3,6]; returns None for the whole document."""
        if not spec:
            return None
        out = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, _, b = part.partition("-")
                out.extend(range(int(a) - 1, int(b)))
            else:
                out.append(int(part) - 1)
        return [p for p in out if 0 <= p < n_pages]

    def _run(self) -> str:
        try:
            import pypdfium2 as pdfium
            from .tex_extract import page_to_tex
        except ImportError as e:
            return self.format_error(
                error="Missing Dependency",
                reason=str(e),
                suggestion="Install the literature bundle: tb install heptapod --bundle literature",
            )

        src = self._safe_path(self.pdf_path)
        rel_out = self.output_path or (os.path.splitext(self.pdf_path)[0] + ".tex")
        dst = self._safe_path(rel_out)

        if not src or not dst:
            return self.format_error(
                error="Access Denied",
                reason="pdf_path or output_path escapes base_directory",
                suggestion="Use relative paths inside base_directory",
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="PDF file not found",
                context=f"path={self.pdf_path}",
                suggestion="Provide a valid PDF path relative to base_directory",
            )

        try:
            doc = pdfium.PdfDocument(src)
            n_pages = len(doc)
            selected = self._parse_pages(self.pages, n_pages)
            indices = range(n_pages) if selected is None else selected
            if selected is not None and not selected:
                return self.format_error(
                    error="Invalid Page Selection",
                    reason=f"no pages matched {self.pages!r}",
                    context=f"document has {n_pages} pages",
                    suggestion="Use 1-based indices within the document length",
                )
            text = "\n".join(page_to_tex(doc[i]) for i in indices)
        except Exception as e:
            return self.format_error(
                error="Extraction Error",
                reason=str(e),
                context=f"path={self.pdf_path}",
                suggestion="Verify the file is a valid PDF; scanned PDFs have no text layer",
            )

        macros = re.findall(r"\\([a-zA-Z]+)", text)
        n_math = text.count("$") // 2
        if not text.strip():
            return self.format_error(
                error="No Text Layer",
                reason="the PDF contains no extractable text",
                context=f"path={self.pdf_path}",
                suggestion="This is likely a scanned document; it needs OCR, which this tool does not do",
            )

        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={rel_out}",
                suggestion="Check that the output directory is writable",
            )

        top = sorted(set(macros), key=lambda m: -macros.count(m))[:12]
        pages_desc = f"{len(list(indices))} of {n_pages}" if selected is not None else str(n_pages)
        return (
            f"Extracted {pages_desc} page(s) to {rel_out}\n"
            f"  schema      : {SCHEMA_VERSION}\n"
            f"  characters  : {len(text):,}\n"
            f"  math runs   : {n_math:,}\n"
            f"  TeX macros  : {len(macros):,} ({len(set(macros))} distinct)\n"
            f"  most common : {', '.join('\\\\' + m for m in top) if top else '(none)'}\n"
            f"\n"
            f"Read the file to inspect the content. Output is TeX-flavoured for "
            f"reading and is not guaranteed to compile: \\mathrm and other upright "
            f"math styles render in the prose font and cannot be recovered, aligned "
            f"environments flatten, and a fraction inside a matrix cell may read as "
            f"an extra row. Where an arXiv LaTeX source exists it remains the "
            f"authoritative version."
        )
