from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from .logging_setup import get_logger
from .detect_regions import Region
from .store_locally import write_packet

log = get_logger("pipeline")

# The rendered PDF page images are large; PIL's decompression-bomb guard
# blocks them by default. We only ever open PDF renders we produced ourselves.
Image.MAX_IMAGE_PIXELS = None


def crop_signature_permissive(
    page_image_path: str,
    band: Region,
    out_path: str,
    padding: int = 20,
) -> dict | None:
    """Same contract as `crop_signature_from_band` but tuned to recover
    signatures the strict version dropped. Differences:
      - Morphological close before projection, so broken strokes count as
        one blob rather than each being filtered as background.
      - Looser projection threshold (0.5% of max), so light flourishes are
        not clipped off the ends.
      - Larger padding around the final crop.
    """
    page = cv2.imread(page_image_path)
    if page is None:
        return None
    H, W = page.shape[:2]
    x0 = max(0, min(W, band.x))
    y0 = max(0, min(H, band.y))
    x1 = max(0, min(W, band.x + band.w))
    y1 = max(0, min(H, band.y + band.h))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    sub = page[y0:y1, x0:x1]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Bridge broken strokes so projections see a single blob per signature.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    if cv2.countNonZero(closed) < 50:
        return None

    col_ink = (closed > 0).sum(axis=0)
    row_ink = (closed > 0).sum(axis=1)
    col_thresh = max(1, int(col_ink.max() * 0.005))
    row_thresh = max(1, int(row_ink.max() * 0.005))

    cols = np.where(col_ink > col_thresh)[0]
    rows = np.where(row_ink > row_thresh)[0]
    if cols.size == 0 or rows.size == 0:
        return None

    cx0, cx1 = int(cols[0]), int(cols[-1]) + 1
    cy0, cy1 = int(rows[0]), int(rows[-1]) + 1
    cx0 = max(0, cx0 - padding)
    cy0 = max(0, cy0 - padding)
    cx1 = min(sub.shape[1], cx1 + padding)
    cy1 = min(sub.shape[0], cy1 + padding)

    crop = sub[cy0:cy1, cx0:cx1]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)

    return {
        "x": int(x0 + cx0),
        "y": int(y0 + cy0),
        "w": int(cx1 - cx0),
        "h": int(cy1 - cy0),
        "method": "ink_projection_permissive",
    }


def detect_page_rules(page_image_path: str) -> tuple[list[int], list[int]]:
    """Detect the printed table grid on a rendered catalogue page.

    Returns (h_ys, v_xs) — sorted pixel positions of horizontal and vertical
    rule lines on the page. Uses HoughLinesP so slight scan skew (up to a
    few degrees) doesn't defeat detection the way strict morphological
    line-opening does. Called once per page so every entry on the page
    snaps to the same rule set."""
    page = cv2.imread(page_image_path, cv2.IMREAD_GRAYSCALE)
    if page is None:
        return [], []
    H, W = page.shape
    edges = cv2.Canny(page, 30, 90)

    min_h_len = max(200, W // 8)
    min_v_len = max(200, H // 8)
    segments = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 720,
        threshold=60,
        minLineLength=200,
        maxLineGap=80,
    )
    if segments is None:
        return [], []

    h_segs: list[tuple[int, int, int]] = []  # (y, x_lo, x_hi)
    v_segs: list[tuple[int, int, int]] = []  # (x, y_lo, y_hi)
    for seg in segments.reshape(-1, 4):
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if abs(dy) <= max(4, abs(dx) * 0.03) and length >= min_h_len:
            h_segs.append((int((y1 + y2) / 2), min(x1, x2), max(x1, x2)))
        elif abs(dx) <= max(4, abs(dy) * 0.03) and length >= min_v_len:
            v_segs.append((int((x1 + x2) / 2), min(y1, y2), max(y1, y2)))

    h_ys = _aggregate_rules(h_segs, page_dim=W, min_coverage=0.4,
                            merge_gap=max(6, H // 200))
    v_xs = _aggregate_rules(v_segs, page_dim=H, min_coverage=0.3,
                            merge_gap=max(6, W // 200))
    return h_ys, v_xs


def _aggregate_rules(
    segs: list[tuple[int, int, int]],
    page_dim: int,
    min_coverage: float,
    merge_gap: int,
) -> list[int]:
    """Cluster line segments by their primary axis, then keep only clusters
    whose combined span (union of segment extents) covers at least
    `min_coverage` of `page_dim`. Filters out signature underlines and other
    short in-cell strokes that Hough returned as line segments."""
    if not segs:
        return []
    segs = sorted(segs)  # by primary-axis position
    clusters: list[list[tuple[int, int, int]]] = [[segs[0]]]
    for s in segs[1:]:
        if s[0] - clusters[-1][-1][0] <= merge_gap:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    kept: list[int] = []
    for cl in clusters:
        spans = sorted((s[1], s[2]) for s in cl)
        total = 0
        cur_a, cur_b = spans[0]
        for a, b in spans[1:]:
            if a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                total += cur_b - cur_a
                cur_a, cur_b = a, b
        total += cur_b - cur_a
        if total / max(1, page_dim) >= min_coverage:
            kept.append(int(round(sum(s[0] for s in cl) / len(cl))))
    return kept


def compute_global_v_xs(page_image_paths: list[str]) -> list[int]:
    """Aggregate v-rule detections across all pages to build a canonical
    column template. The table's vertical rules are printed at identical
    x-positions on every page; pooling detections recovers rules that
    slipped through on individual pages (typically due to scan skew that
    made some verticals fall below the per-page detection threshold).

    Spurious per-page detections (e.g. a signature descender that survived
    a page's HoughLinesP filter) show up as low-frequency clusters. We
    require a rule position to be present on at least 25% of pages to
    survive."""
    per_page: list[list[int]] = []
    for p in page_image_paths:
        _, v = detect_page_rules(p)
        per_page.append(v)
    all_v = [x for page in per_page for x in page]
    if not all_v:
        return []
    clusters = _cluster_line_positions(np.array(sorted(all_v)), gap=30)
    n_pages = max(1, len(per_page))
    threshold = max(2, int(n_pages * 0.25))
    kept: list[int] = []
    for c in clusters:
        hits = sum(1 for page in per_page
                   if any(abs(x - c) <= 30 for x in page))
        if hits >= threshold:
            kept.append(c)
    return kept


def crop_signature_to_cell(
    page_image_path: str,
    band: Region,
    h_ys: list[int],
    v_xs: list[int],
    out_path: str,
    inset: int = 4,
    median_row_h: int | None = None,
) -> dict | None:
    """Snap `band` to the printed table cell that contains its center by
    picking the nearest bracketing horizontal + vertical rules on each side.
    Returns a page-coord bbox dict (with `method="page_grid"`), or None if
    no complete set of bracketing rules exists.

    If `median_row_h` is supplied, missing top or bottom horizontal brackets
    (typical for the first/last row on a page) are synthesized from it so
    edge rows still snap to a cell-sized crop."""
    if not v_xs:
        return None
    page = cv2.imread(page_image_path)
    if page is None:
        return None
    H, W = page.shape[:2]

    cx = band.x + band.w // 2
    cy = band.y + band.h // 2
    above = [y for y in h_ys if y < cy]
    below = [y for y in h_ys if y > cy]
    left_v = [x for x in v_xs if x < cx]
    right_v = [x for x in v_xs if x > cx]
    if not (left_v and right_v):
        return None
    left = max(left_v)
    right = min(right_v)

    if above and below:
        top = max(above)
        bottom = min(below)
    elif above and median_row_h:
        top = max(above)
        bottom = min(H, top + median_row_h)
    elif below and median_row_h:
        bottom = min(below)
        top = max(0, bottom - median_row_h)
    else:
        return None

    if bottom - top < 30 or right - left < 60:
        return None

    cx0 = max(0, left + inset)
    cy0 = max(0, top + inset)
    cx1 = min(W, right - inset)
    cy1 = min(H, bottom - inset)
    if cx1 - cx0 < 20 or cy1 - cy0 < 20:
        return None

    crop = page[cy0:cy1, cx0:cx1]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
    return {
        "x": int(cx0),
        "y": int(cy0),
        "w": int(cx1 - cx0),
        "h": int(cy1 - cy0),
        "method": "page_grid",
    }


def _cluster_line_positions(indices: np.ndarray, gap: int = 4) -> list[int]:
    """Collapse runs of consecutive line-row/col indices into single positions."""
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for i in indices[1:]:
        if int(i) - groups[-1][-1] <= gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [int(round(sum(g) / len(g))) for g in groups]


def _find_enclosing_box(sub: np.ndarray) -> tuple[int, int, int, int] | None:
    """Snap to the printed cell rectangle surrounding a signature specimen.

    The catalogue pages are drawn as a grid of thin horizontal and vertical
    rule lines. Contour detection fails because every cell shares its border
    with its neighbours (one giant connected contour). Instead we detect the
    rule *lines* themselves via morphological opening with long horizontal /
    vertical kernels, then pick the pair of horizontals that bracket the
    vertical centre of `sub` and the pair of verticals that bracket its
    horizontal centre. Returns (x,y,w,h) in `sub` coords, or None if no
    convincing bracketing lines were found on all four sides.
    """
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    if H < 30 or W < 60:
        return None

    _, bw = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Long lines only: kernel length ~75% of the corresponding dimension so
    # ink strokes (letters, flourishes, underlines) are washed out and only
    # page-spanning rule lines survive.
    h_len = max(40, int(W * 0.55))
    v_len = max(30, int(H * 0.55))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    # A row that is >=40% covered by horizontal-line ink is a rule row.
    h_row_cov = (h_lines > 0).sum(axis=1) / max(1, W)
    v_col_cov = (v_lines > 0).sum(axis=0) / max(1, H)
    h_positions = _cluster_line_positions(np.where(h_row_cov > 0.5)[0])
    v_positions = _cluster_line_positions(np.where(v_col_cov > 0.5)[0])

    cy_mid, cx_mid = H // 2, W // 2

    above = [p for p in h_positions if p < cy_mid - 4]
    below = [p for p in h_positions if p > cy_mid + 4]
    left_lines = [p for p in v_positions if p < cx_mid - 4]
    right_lines = [p for p in v_positions if p > cx_mid + 4]

    if not (above and below and left_lines and right_lines):
        return None

    # Use the outermost bracketing lines. Real page rules are the outer ones;
    # anything closer to the ink centre is either a stroke that survived the
    # long-line filter or an underline inside the signature — either way we
    # do not want to clip on it.
    top = min(above)
    bottom = max(below)
    left = min(left_lines)
    right = max(right_lines)

    box_w = right - left
    box_h = bottom - top
    if box_w < 40 or box_h < 20:
        return None

    return left, top, box_w, box_h


def crop_signature_from_band(
    page_image_path: str,
    band: Region,
    out_path: str,
    padding: int = 12,
    border_inset: int = 4,
) -> dict | None:
    """`band` is the search rectangle (right column) in page-pixel coords.
    If a printed rectangular border is visible inside the band, crop to that
    border (inset by `border_inset` px so the border line itself is excluded).
    Otherwise fall back to the tightest ink-projection bbox. Returns the crop
    bbox in page-pixel coords with a `method` key indicating which path was
    used, or None if the band is empty.
    """
    page = cv2.imread(page_image_path)
    if page is None:
        log.warning(f"could not read {page_image_path}")
        return None

    H, W = page.shape[:2]
    x0 = max(0, min(W, band.x))
    y0 = max(0, min(H, band.y))
    x1 = max(0, min(W, band.x + band.w))
    y1 = max(0, min(H, band.y + band.h))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    # Expand the search area so rule lines that sit at (or just outside)
    # the band's edges — very common, because the band is derived from text
    # extents that abut the cell borders — fall inside the search window.
    margin = 120
    ex0 = max(0, x0 - margin)
    ey0 = max(0, y0 - margin)
    ex1 = min(W, x1 + margin)
    ey1 = min(H, y1 + margin)
    expanded = page[ey0:ey1, ex0:ex1]

    box = _find_enclosing_box(expanded)
    if box is not None:
        bx, by, bw, bh = box
        cx0 = max(0, bx + border_inset)
        cy0 = max(0, by + border_inset)
        cx1 = min(expanded.shape[1], bx + bw - border_inset)
        cy1 = min(expanded.shape[0], by + bh - border_inset)
        if cx1 - cx0 >= 20 and cy1 - cy0 >= 20:
            crop = expanded[cy0:cy1, cx0:cx1]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
            return {
                "x": int(ex0 + cx0),
                "y": int(ey0 + cy0),
                "w": int(cx1 - cx0),
                "h": int(cy1 - cy0),
                "method": "enclosing_box",
            }

    sub = page[y0:y1, x0:x1]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    # Otsu binarisation with inverse so ink = white.
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ink = cv2.countNonZero(bw)
    if ink < 50:
        log.info(f"{Path(page_image_path).name}: empty band at "
                 f"({band.x},{band.y},{band.w},{band.h})")
        return None

    # Tighten: shave off rows/cols that are mostly background so the crop
    # hugs the ink. Use column/row projections.
    col_ink = (bw > 0).sum(axis=0)
    row_ink = (bw > 0).sum(axis=1)
    col_thresh = max(1, int(col_ink.max() * 0.02))
    row_thresh = max(1, int(row_ink.max() * 0.02))

    cols = np.where(col_ink > col_thresh)[0]
    rows = np.where(row_ink > row_thresh)[0]
    if cols.size == 0 or rows.size == 0:
        return None

    cx0, cx1 = int(cols[0]), int(cols[-1]) + 1
    cy0, cy1 = int(rows[0]), int(rows[-1]) + 1

    cx0 = max(0, cx0 - padding)
    cy0 = max(0, cy0 - padding)
    cx1 = min(sub.shape[1], cx1 + padding)
    cy1 = min(sub.shape[0], cy1 + padding)

    crop = sub[cy0:cy1, cx0:cx1]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)

    return {
        "x": int(x0 + cx0),
        "y": int(y0 + cy0),
        "w": int(cx1 - cx0),
        "h": int(cy1 - cy0),
        "method": "ink_projection",
    }


def repair_outlier_crops(workdir: Path, v_xs_global: list[int]) -> int:
    """Second-pass repair: any crop whose width or height is far below the
    per-page median (i.e., clearly not the whole cell) gets re-cropped
    using the median cell dimensions, centred on its original search band.

    Returns the number of entries repaired. Entries are updated in place
    and their packets regenerated."""
    entries_dir = workdir / "entries"
    crops_dir = workdir / "crops"
    pages_dir = workdir / "pages"

    by_page: dict[int, list[dict]] = {}
    for p in sorted(entries_dir.glob("*.json")):
        e = json.loads(p.read_text())
        by_page.setdefault(e["page_number"], []).append(e)

    # Global column bounds — the same for every page in the PDF.
    global_left = min(v_xs_global) if v_xs_global else None
    global_right = max(v_xs_global) if v_xs_global else None

    # Collect neighbour siblings on each page — for a bad entry we can snap
    # its top/bottom to the neighbouring entry's grid-cropped bounds, which
    # are the actual cell borders.
    repaired = 0
    for page_num, page_entries in by_page.items():
        # Sort by page-y so we can find above/below neighbours.
        page_entries.sort(key=lambda e: e["signature_bbox"]["y"])
        good = [e for e in page_entries
                if e.get("crop_method") == "page_grid"]
        ws = [e["signature_bbox"]["w"] for e in good] or \
             [e["signature_bbox"]["w"] for e in page_entries]
        hs = [e["signature_bbox"]["h"] for e in good] or \
             [e["signature_bbox"]["h"] for e in page_entries]
        med_w = int(sorted(ws)[len(ws) // 2])
        med_h = int(sorted(hs)[len(hs) // 2])
        # A "good" entry on this page has w >= 0.6*median_w AND
        # h >= 0.5*median_h; anything below either bar is a repair target.
        w_thr = med_w * 0.6
        h_thr = med_h * 0.5

        page_path = pages_dir / f"page_{page_num:04d}.png"
        if not page_path.exists():
            continue
        page = cv2.imread(str(page_path))
        if page is None:
            continue
        H, W = page.shape[:2]

        for idx, e in enumerate(page_entries):
            b = e["signature_bbox"]
            if b["w"] >= w_thr and b["h"] >= h_thr:
                continue

            band = e["signature_search_bbox"]

            # Horizontal bounds: signatures always live in the rightmost
            # column, bounded by the last two verticals in the global
            # template (text/sig divider on the left, outer border on the
            # right). This is more reliable than picking rules by proximity
            # to the search band, which can drift into the wrong column
            # when the text extraction misjudged the entry's right edge.
            if v_xs_global and len(v_xs_global) >= 2:
                left = v_xs_global[-2]
                right = v_xs_global[-1]
            else:
                cx = band["x"] + band["w"] // 2
                left = cx - med_w // 2
                right = cx + med_w // 2

            # Vertical bounds: prefer the neighbouring grid-cropped cells'
            # boundaries. The cell above's bottom = this cell's top, etc.
            top = None
            for prev in reversed(page_entries[:idx]):
                if prev.get("crop_method") == "page_grid":
                    pb = prev["signature_bbox"]
                    top = pb["y"] + pb["h"]
                    break
            bottom = None
            for nxt in page_entries[idx + 1:]:
                if nxt.get("crop_method") == "page_grid":
                    nb = nxt["signature_bbox"]
                    bottom = nb["y"]
                    break

            # Fall back to the search band's own y-range — always inside
            # the cell (since it was derived from text extents in that row)
            # even if not the full cell height.
            if top is None:
                top = band["y"]
            if bottom is None:
                bottom = band["y"] + band["h"]

            # Clamp to a plausible cell height so a runaway span (e.g. very
            # short entry above, tall gap) doesn't produce a giant crop.
            if bottom - top > med_h * 1.6:
                cy = band["y"] + band["h"] // 2
                top = max(top, cy - med_h)
                bottom = min(bottom, cy + med_h)

            inset = 4
            cx0 = max(0, left + inset)
            cy0 = max(0, top + inset)
            cx1 = min(W, right - inset)
            cy1 = min(H, bottom - inset)
            if cx1 - cx0 < 20 or cy1 - cy0 < 20:
                continue

            crop = page[cy0:cy1, cx0:cx1]
            out_path = crops_dir / f"{e['id']}.png"
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(out_path)
            e["signature_bbox"] = {
                "x": cx0, "y": cy0, "w": cx1 - cx0, "h": cy1 - cy0,
            }
            e["crop_method"] = "repaired_median"
            (entries_dir / f"{e['id']}.json").write_text(json.dumps(e, indent=2))
            write_packet(workdir, e)
            repaired += 1

    return repaired
