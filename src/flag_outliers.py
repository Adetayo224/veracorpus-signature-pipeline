from __future__ import annotations
import json
from pathlib import Path
from statistics import median

from .logging_setup import get_logger
from .store_locally import write_packet

log = get_logger("pipeline")


# Tolerance bands (multiplicative around the per-PDF median).
AREA_SMALL = 0.25
AREA_LARGE = 4.0
DIM_TIGHT = 0.4
DIM_LOOSE = 2.5


def _classify(w: int, h: int,
              med_w: float, med_h: float, med_area: float) -> str:
    area = w * h
    if area < med_area * AREA_SMALL:
        return "too_small"
    if area > med_area * AREA_LARGE:
        return "too_large"
    if w < med_w * DIM_TIGHT:
        return "too_narrow"
    if w > med_w * DIM_LOOSE:
        return "too_wide"
    if h < med_h * DIM_TIGHT:
        return "too_short"
    if h > med_h * DIM_LOOSE:
        return "too_tall"
    return "ok"


def flag_pdf(pdf_dir: Path) -> dict:
    """Scan one processed/<pdf>/ folder, tag each entry with `crop_quality`,
    rewrite its packet, and return a summary dict."""
    ent_dir = pdf_dir / "entries"
    if not ent_dir.exists():
        return {"pdf": pdf_dir.name, "total": 0, "flagged": 0, "by_class": {}}

    entries: list[tuple[Path, dict]] = []
    for p in sorted(ent_dir.glob("*.json")):
        try:
            entries.append((p, json.loads(p.read_text())))
        except Exception as e:
            log.warning(f"skip unreadable entry {p}: {e}")

    dims = []
    for _, e in entries:
        bb = e.get("signature_bbox") or {}
        w, h = int(bb.get("w", 0)), int(bb.get("h", 0))
        if w > 0 and h > 0:
            dims.append((w, h))
    if not dims:
        return {"pdf": pdf_dir.name, "total": len(entries),
                "flagged": 0, "by_class": {}}

    med_w = median(w for w, _ in dims)
    med_h = median(h for _, h in dims)
    med_area = median(w * h for w, h in dims)

    by_class: dict[str, int] = {}
    for path, entry in entries:
        bb = entry.get("signature_bbox") or {}
        w, h = int(bb.get("w", 0)), int(bb.get("h", 0))
        cls = _classify(w, h, med_w, med_h, med_area) if (w and h) else "empty"
        entry["crop_quality"] = cls
        entry["crop_quality_ref"] = {
            "median_w": round(med_w, 1),
            "median_h": round(med_h, 1),
            "median_area": round(med_area, 1),
        }
        path.write_text(json.dumps(entry, indent=2))
        write_packet(pdf_dir, entry)
        by_class[cls] = by_class.get(cls, 0) + 1

    flagged = sum(v for k, v in by_class.items() if k != "ok")
    return {
        "pdf": pdf_dir.name,
        "total": len(entries),
        "flagged": flagged,
        "median_w": round(med_w, 1),
        "median_h": round(med_h, 1),
        "median_area": round(med_area, 1),
        "by_class": by_class,
    }


def flag_all(processed_root: str = "processed") -> list[dict]:
    root = Path(processed_root)
    if not root.exists():
        return []
    reports = []
    for pdf_dir in sorted(root.iterdir()):
        if not pdf_dir.is_dir():
            continue
        r = flag_pdf(pdf_dir)
        (pdf_dir / "quality_report.json").write_text(json.dumps(r, indent=2))
        reports.append(r)
        log.info(f"{r['pdf']}: {r['total']} entries, "
                 f"{r['flagged']} flagged — {r['by_class']}")
    return reports
