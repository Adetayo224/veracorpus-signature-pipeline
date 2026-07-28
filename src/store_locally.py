from __future__ import annotations
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path

from .logging_setup import get_logger

log = get_logger("pipeline")


def pdf_workdir(pdf_path: str, base: str = "processed") -> Path:
    stem = Path(pdf_path).stem
    d = Path(base) / stem
    d.mkdir(parents=True, exist_ok=True)
    (d / "pages").mkdir(exist_ok=True)
    (d / "crops").mkdir(exist_ok=True)
    (d / "entries").mkdir(exist_ok=True)
    (d / "packets").mkdir(exist_ok=True)
    return d


def write_entry(workdir: Path, entry: dict) -> str:
    entry_id = entry.get("id") or str(uuid.uuid4())[:8]
    entry["id"] = entry_id
    entry.setdefault("review_status", "pending")
    entry.setdefault("pushed", False)
    entry.setdefault("created_at", datetime.utcnow().isoformat())
    path = workdir / "entries" / f"{entry_id}.json"
    path.write_text(json.dumps(entry, indent=2))
    write_packet(workdir, entry)
    return entry_id


def write_packet(workdir: Path, entry: dict) -> Path:
    """Write a single self-contained document per entry that bundles the OCR
    metadata + the cropped signature PNG (base64-embedded). This is the
    atomic 'ready-to-push' unit consumed by the future Supabase push module.
    Regenerated on every entry write and on every review-UI edit."""
    entry_id = entry["id"]
    crop_path = workdir / "crops" / f"{entry_id}.png"
    packet = dict(entry)
    if crop_path.exists():
        packet["signature_image"] = {
            "filename": f"{entry_id}.png",
            "mime_type": "image/png",
            "encoding": "base64",
            "data": base64.b64encode(crop_path.read_bytes()).decode("ascii"),
        }
    else:
        packet["signature_image"] = None
    packet["bundled_at"] = datetime.utcnow().isoformat()
    out = workdir / "packets" / f"{entry_id}.json"
    out.write_text(json.dumps(packet, indent=2))
    return out


def write_manifest(workdir: Path, source_pdf: str) -> None:
    entries = []
    for p in sorted((workdir / "entries").glob("*.json")):
        try:
            entries.append(json.loads(p.read_text()))
        except Exception as e:
            log.warning(f"Bad entry file {p}: {e}")
    manifest = {
        "source_pdf": source_pdf,
        "generated_at": datetime.utcnow().isoformat(),
        "entry_count": len(entries),
        "entries": [
            {
                "id": e["id"],
                "page": e.get("page_number"),
                "review_status": e.get("review_status"),
                "pushed": e.get("pushed", False),
            }
            for e in entries
        ],
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
