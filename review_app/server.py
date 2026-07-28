from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

Image.MAX_IMAGE_PIXELS = None  # our page renders can exceed PIL's default cap

APP_DIR = Path(__file__).parent
PROCESSED = Path("processed").resolve()
PROCESSED.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VeraCorpus Signature Review")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/processed", StaticFiles(directory=PROCESSED), name="processed")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def _pdf_dirs() -> list[Path]:
    return sorted([p for p in PROCESSED.iterdir() if p.is_dir()]) \
        if PROCESSED.exists() else []


def _load_entries(pdf_dir: Path) -> list[dict]:
    ent_dir = pdf_dir / "entries"
    if not ent_dir.exists():
        return []
    entries = []
    for p in sorted(ent_dir.glob("*.json")):
        try:
            entries.append(json.loads(p.read_text()))
        except Exception:
            continue
    return entries


def _save_entry(pdf_dir: Path, entry: dict) -> None:
    (pdf_dir / "entries" / f"{entry['id']}.json").write_text(
        json.dumps(entry, indent=2))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    summaries = []
    for d in _pdf_dirs():
        entries = _load_entries(d)
        summaries.append({
            "name": d.name,
            "total": len(entries),
            "pending": sum(1 for e in entries if e.get("review_status") == "pending"),
            "accepted": sum(1 for e in entries if e.get("review_status") == "accepted"),
            "edited": sum(1 for e in entries if e.get("review_status") == "edited"),
            "pushed": sum(1 for e in entries if e.get("pushed")),
        })
    return templates.TemplateResponse(
        request, "index.html", {"summaries": summaries})


@app.get("/pdf/{pdf_name}", response_class=HTMLResponse)
def pdf_view(pdf_name: str, request: Request):
    pdf_dir = PROCESSED / pdf_name
    if not pdf_dir.exists():
        raise HTTPException(404, "PDF not found")
    entries = _load_entries(pdf_dir)
    return templates.TemplateResponse(
        request, "pdf.html",
        {"pdf_name": pdf_name, "entries": entries},
    )


class AcceptBody(BaseModel):
    artist_name_claimed: str | None = None
    dates: str | None = None
    title: str | None = None


@app.post("/entry/{pdf_name}/{entry_id}/accept")
def accept(pdf_name: str, entry_id: str, body: AcceptBody):
    pdf_dir = PROCESSED / pdf_name
    path = pdf_dir / "entries" / f"{entry_id}.json"
    if not path.exists():
        raise HTTPException(404, "entry not found")
    entry = json.loads(path.read_text())
    for k, v in body.model_dump(exclude_none=True).items():
        entry[k] = v
    entry["review_status"] = "accepted"
    _save_entry(pdf_dir, entry)
    return {"ok": True, "entry": entry}


class EditBody(BaseModel):
    x: int
    y: int
    w: int
    h: int
    artist_name_claimed: str | None = None
    dates: str | None = None
    title: str | None = None


@app.post("/entry/{pdf_name}/{entry_id}/edit")
def edit(pdf_name: str, entry_id: str, body: EditBody):
    pdf_dir = PROCESSED / pdf_name
    path = pdf_dir / "entries" / f"{entry_id}.json"
    if not path.exists():
        raise HTTPException(404, "entry not found")
    entry = json.loads(path.read_text())
    page_no = int(entry.get("page_number", 0))
    page_path = pdf_dir / "pages" / f"page_{page_no:04d}.png"
    if not page_path.exists():
        raise HTTPException(500, f"page image missing: {page_path.name}")

    img = Image.open(page_path)
    x0 = max(0, body.x)
    y0 = max(0, body.y)
    x1 = min(img.width, body.x + body.w)
    y1 = min(img.height, body.y + body.h)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(400, "invalid crop rectangle")
    crop = img.crop((x0, y0, x1, y1))
    crop_path = pdf_dir / "crops" / f"{entry_id}.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(crop_path)

    entry["signature_bbox"] = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
    entry["crop_method"] = "manual_edit"
    for k in ("artist_name_claimed", "dates", "title"):
        v = getattr(body, k)
        if v is not None:
            entry[k] = v
    entry["review_status"] = "edited"
    _save_entry(pdf_dir, entry)
    return {"ok": True, "entry": entry}


@app.post("/push")
def push():
    from src.push_to_supabase import push_all
    n = push_all()
    return {"pushed": n}
