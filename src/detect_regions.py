from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF

from .logging_setup import get_logger

log = get_logger("pipeline")


@dataclass
class Region:
    """A rectangle in *rendered-pixel* coordinates for a given page image."""
    kind: str  # 'image_region' | 'text_region'
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PageEntry:
    """One catalogue entry on a page: a text block on the left and the
    corresponding signature band on the right. Coordinates are in rendered
    pixels at the current render DPI."""
    page_index: int  # 0-based
    text_region: Region
    signature_region: Region
    artist_name_claimed: str
    dates: str
    title: str
    raw_text: str

    def to_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "text_region": self.text_region.to_dict(),
            "signature_region": self.signature_region.to_dict(),
            "artist_name_claimed": self.artist_name_claimed,
            "dates": self.dates,
            "title": self.title,
            "raw_text": self.raw_text,
        }


# --- text-block grouping -----------------------------------------------------

def _is_header_line(line: str) -> bool:
    """True if `line` looks like a catalogue entry header (surname).
    Headers are uppercase names like 'AACHEN', 'ABBEY, RA', 'ACHENBA CH'.
    Rule: at least 3 alphabetic characters, and every alphabetic character
    is uppercase (letters only — accented A-Z included via isalpha)."""
    s = line.strip()
    if len(s) < 3:
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return False
    return all(c.isupper() for c in letters)


def _cluster_blocks_into_entries(blocks: list[tuple]) -> list[list[tuple]]:
    """Blocks come from PyMuPDF as (x0,y0,x1,y1,text,bno,btype). Group adjacent
    blocks into per-artist entries.

    Primary split signal: a block whose first non-empty line is an ALL-CAPS
    header (the artist surname) starts a new entry. This is the layout
    convention in these catalogues and is far more reliable than any gap
    heuristic (within-entry gaps can equal between-entry gaps).

    Fallback: if no headers appear on the page, split on unusually large
    vertical gaps."""
    if not blocks:
        return []
    text_blocks = sorted(
        [b for b in blocks if b[6] == 0 and b[4].strip()],
        key=lambda b: b[1],
    )
    if not text_blocks:
        return []

    def first_line(b: tuple) -> str:
        for ln in b[4].splitlines():
            s = ln.strip()
            if s:
                return s
        return ""

    header_flags = [_is_header_line(first_line(b)) for b in text_blocks]

    if any(header_flags):
        clusters: list[list[tuple]] = []
        current: list[tuple] = []
        for b, is_header in zip(text_blocks, header_flags):
            if is_header and current:
                clusters.append(current)
                current = [b]
            else:
                current.append(b)
        if current:
            clusters.append(current)
        # If a page starts with continuation text before the first header,
        # drop that stray leading cluster (usually a running header/footer).
        if clusters and not _is_header_line(first_line(clusters[0][0])):
            clusters = clusters[1:] or clusters
        return clusters

    # Fallback for header-less pages: gap-based split.
    gaps = [max(0, n[1] - p[3]) for p, n in zip(text_blocks, text_blocks[1:])]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
    split_thresh = max(60.0, median_gap * 3.5)
    clusters = [[text_blocks[0]]]
    for prev, nxt in zip(text_blocks, text_blocks[1:]):
        if nxt[1] - prev[3] > split_thresh:
            clusters.append([nxt])
        else:
            clusters[-1].append(nxt)
    return clusters


def _parse_entry_text(cluster: list[tuple]) -> tuple[str, str, str, str]:
    """Return (artist_name, dates, title, raw_text) from a cluster of blocks."""
    import re

    parts: list[str] = []
    for b in cluster:
        parts.append(b[4].strip())
    raw = "\n".join(parts).strip()

    # First non-empty line of the first block is typically the surname header.
    first_block_text = cluster[0][4].strip()
    first_lines = [ln.strip() for ln in first_block_text.splitlines() if ln.strip()]
    artist = first_lines[0] if first_lines else ""
    # Sometimes the given name lives on subsequent lines of the same block or
    # in a second small block; append the next line(s) until we hit something
    # that looks like biography (contains "b." or a 4-digit year).
    year_re = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
    extra = []
    for line in first_lines[1:]:
        if year_re.search(line) or line.lower().startswith("b."):
            break
        extra.append(line)
    if extra:
        artist = f"{artist} {' '.join(extra)}".strip()

    dates = ", ".join(year_re.findall(raw))

    # Title: heuristic — pick a short descriptive phrase between the bio and
    # the "Exh:" section, if present.
    title = ""
    for line in raw.splitlines():
        s = line.strip()
        if s.lower().startswith("exh:"):
            break
        if year_re.search(s) or s.lower().startswith("b."):
            continue
        if s and s != artist and len(s) < 80:
            title = s
            break

    return artist, dates, title, raw


# --- signature band placement ------------------------------------------------

def _signature_band_for_cluster(
    cluster: list[tuple],
    page_width_pt: float,
    page_height_pt: float,
    next_cluster_top: float | None,
    scale: float,
) -> Region:
    """The signature specimen sits to the right of the text at the same y-band.
    We take a rectangle from just right of the text column to the right page
    margin, spanning from this cluster's top to the top of the next cluster
    (or the page bottom, minus a footer margin)."""
    y0 = min(b[1] for b in cluster)
    y1_text = max(b[3] for b in cluster)
    y1 = next_cluster_top - 10 if next_cluster_top is not None else y1_text + (page_height_pt * 0.05)
    y1 = max(y1, y1_text)  # never smaller than the text itself
    y1 = min(y1, page_height_pt - 20)

    text_right = max(b[2] for b in cluster)
    x0 = text_right + 20
    x1 = page_width_pt - 20

    # Clamp and scale to pixel coords
    x0 = max(0.0, min(page_width_pt, x0))
    x1 = max(0.0, min(page_width_pt, x1))
    y0 = max(0.0, min(page_height_pt, y0))
    y1 = max(0.0, min(page_height_pt, y1))

    return Region(
        kind="image_region",
        x=int(round(x0 * scale)),
        y=int(round(y0 * scale)),
        w=int(round((x1 - x0) * scale)),
        h=int(round((y1 - y0) * scale)),
    )


def _text_region_for_cluster(cluster: list[tuple], scale: float) -> Region:
    x0 = min(b[0] for b in cluster)
    y0 = min(b[1] for b in cluster)
    x1 = max(b[2] for b in cluster)
    y1 = max(b[3] for b in cluster)
    return Region(
        kind="text_region",
        x=int(round(x0 * scale)),
        y=int(round(y0 * scale)),
        w=int(round((x1 - x0) * scale)),
        h=int(round((y1 - y0) * scale)),
    )


# --- public API --------------------------------------------------------------

def detect_page_entries(pdf_path: str, page_index: int, dpi: int) -> list[PageEntry]:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        blocks = page.get_text("blocks")
        clusters = _cluster_blocks_into_entries(blocks)
        scale = dpi / 72.0
        w_pt, h_pt = page.rect.width, page.rect.height
    finally:
        doc.close()

    entries: list[PageEntry] = []
    for i, cluster in enumerate(clusters):
        artist, dates, title, raw = _parse_entry_text(cluster)
        if not artist and not raw:
            continue
        next_top = min(b[1] for b in clusters[i + 1]) if i + 1 < len(clusters) else None
        sig_region = _signature_band_for_cluster(
            cluster, w_pt, h_pt, next_top, scale
        )
        # Ignore entries where the signature band is too small to hold anything
        if sig_region.w < 40 or sig_region.h < 40:
            continue
        txt_region = _text_region_for_cluster(cluster, scale)
        entries.append(PageEntry(
            page_index=page_index,
            text_region=txt_region,
            signature_region=sig_region,
            artist_name_claimed=artist,
            dates=dates,
            title=title,
            raw_text=raw,
        ))

    log.info(f"{Path(pdf_path).name} p{page_index + 1}: "
             f"{len(entries)} entries from {len(clusters)} text clusters")
    return entries
