from __future__ import annotations
import json
from pathlib import Path

import cv2
from PIL import Image

from .logging_setup import get_logger
from .detect_regions import Region
from .crop_signature import crop_signature_permissive
from .store_locally import write_packet
from .flag_outliers import _classify

log = get_logger("pipeline")

Image.MAX_IMAGE_PIXELS = None

# Only retry these classes; leave `ok` alone, and `empty` means we had no
# ink at all in the band — a wider band usually won't help.
RETRY_CLASSES = {"too_small", "too_narrow", "too_short",
                 "too_tall", "too_wide", "too_large"}


def _page_size(page_image_path: Path) -> tuple[int, int]:
    im = cv2.imread(str(page_image_path))
    if im is None:
        return 0, 0
    return im.shape[1], im.shape[0]  # w, h


def _expanded_band(
    original_search: dict,
    page_w: int,
    page_h: int,
    med_w: float,
    med_h: float,
) -> Region:
    """Widen the original search bbox so it can accommodate a median-sized
    signature. Anchor the left edge at a fixed column boundary (30% of the
    page width) rather than the text_right of this specific entry — that
    frees narrow-search cases where the artist description ran wide."""
    left_col = int(page_w * 0.30)
    x0 = min(original_search["x"], left_col)
    x1 = page_w - 20

    # Ensure the band is at least median_w wide.
    if x1 - x0 < med_w * 1.1:
        x0 = max(0, x1 - int(med_w * 1.1))

    y_center = original_search["y"] + original_search["h"] // 2
    target_h = max(int(med_h * 1.4), original_search["h"])
    y0 = max(0, y_center - target_h // 2)
    y1 = min(page_h, y0 + target_h)

    return Region(
        kind="image_region",
        x=int(x0), y=int(y0),
        w=int(x1 - x0), h=int(y1 - y0),
    )


def _score(bbox: dict, med_w: float, med_h: float, med_area: float) -> float:
    """Lower is better: how far this crop deviates from the per-PDF median,
    in a log-ratio sense so being 2x or 0.5x median counts the same."""
    import math
    w, h = bbox["w"], bbox["h"]
    if w <= 0 or h <= 0:
        return float("inf")
    area = w * h
    return (abs(math.log(w / med_w))
            + abs(math.log(h / med_h))
            + abs(math.log(area / med_area)))


def recrop_pdf(pdf_dir: Path) -> dict:
    ent_dir = pdf_dir / "entries"
    pages_dir = pdf_dir / "pages"
    crops_dir = pdf_dir / "crops"
    if not ent_dir.exists():
        return {"pdf": pdf_dir.name, "retried": 0, "improved": 0,
                "kept_original": 0, "still_bad": 0}

    entries: list[tuple[Path, dict]] = []
    for p in sorted(ent_dir.glob("*.json")):
        try:
            entries.append((p, json.loads(p.read_text())))
        except Exception as e:
            log.warning(f"skip unreadable entry {p}: {e}")
    if not entries:
        return {"pdf": pdf_dir.name, "retried": 0, "improved": 0,
                "kept_original": 0, "still_bad": 0}

    ref = entries[0][1].get("crop_quality_ref") or {}
    med_w = float(ref.get("median_w") or 0)
    med_h = float(ref.get("median_h") or 0)
    med_area = float(ref.get("median_area") or 0)
    if med_w <= 0 or med_h <= 0 or med_area <= 0:
        log.warning(f"{pdf_dir.name}: no median reference; run --flag first")
        return {"pdf": pdf_dir.name, "retried": 0, "improved": 0,
                "kept_original": 0, "still_bad": 0}

    page_cache: dict[int, tuple[int, int]] = {}
    retried = improved = kept_original = still_bad = 0

    for path, entry in entries:
        if entry.get("crop_quality") not in RETRY_CLASSES:
            continue
        retried += 1
        page_no = int(entry.get("page_number", 0))
        page_path = pages_dir / f"page_{page_no:04d}.png"
        if not page_path.exists():
            still_bad += 1
            continue
        if page_no not in page_cache:
            page_cache[page_no] = _page_size(page_path)
        page_w, page_h = page_cache[page_no]
        if page_w == 0:
            still_bad += 1
            continue

        original_search = entry["signature_search_bbox"]
        band = _expanded_band(original_search, page_w, page_h, med_w, med_h)

        crop_path = crops_dir / f"{entry['id']}.png"
        # Save the strict crop first (in case we need to revert) — do this
        # by cropping to a temp path, then swapping only on improvement.
        tmp_path = crops_dir / f".tmp_{entry['id']}.png"
        new_bbox = crop_signature_permissive(
            str(page_path), band, str(tmp_path)
        )
        if new_bbox is None:
            if tmp_path.exists():
                tmp_path.unlink()
            still_bad += 1
            continue

        new_method = new_bbox.pop("method", "ink_projection_permissive")
        old_bbox = entry.get("signature_bbox") or {"w": 0, "h": 0}
        old_score = _score(old_bbox, med_w, med_h, med_area)
        new_score = _score(new_bbox, med_w, med_h, med_area)

        if new_score + 0.05 < old_score:
            # Meaningful improvement: keep it.
            tmp_path.replace(crop_path)
            entry["signature_bbox"] = new_bbox
            entry["signature_search_bbox"] = band.to_dict()
            entry["crop_method"] = new_method
            entry["recropped"] = True
            new_cls = _classify(new_bbox["w"], new_bbox["h"],
                                med_w, med_h, med_area)
            entry["crop_quality"] = new_cls
            path.write_text(json.dumps(entry, indent=2))
            write_packet(pdf_dir, entry)
            improved += 1
            if new_cls != "ok":
                still_bad += 1
        else:
            tmp_path.unlink(missing_ok=True)
            kept_original += 1
            still_bad += 1

    return {
        "pdf": pdf_dir.name,
        "retried": retried,
        "improved": improved,
        "kept_original": kept_original,
        "still_bad": still_bad,
    }


def recrop_all(processed_root: str = "processed") -> list[dict]:
    root = Path(processed_root)
    if not root.exists():
        return []
    reports = []
    for pdf_dir in sorted(root.iterdir()):
        if not pdf_dir.is_dir():
            continue
        r = recrop_pdf(pdf_dir)
        reports.append(r)
        log.info(
            f"{r['pdf']}: retried {r['retried']}, improved {r['improved']}, "
            f"kept {r['kept_original']}, still_bad {r['still_bad']}"
        )
    return reports
