# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VeraCorpus Signature Extraction and Review Pipeline — a local desktop pipeline that ingests multi-page PDF catalogues of sculpture entries, auto-detects and crops signature/inscription regions, OCRs the surrounding descriptive text, presents results in a local review UI for accept/re-crop, and on command pushes approved entries to Supabase.

Performance target: a 60-page PDF must finish the local detection + cropping stage in well under one minute (excluding manual review and Gemini API calls).

## Commands

Environment lives in `.venv/` at the project root.

```bash
# Activate
source .venv/bin/activate

# Install / update deps
pip install -r requirements.txt

# Also needs the Tesseract binary on the system
# (pytesseract wraps it): apt install tesseract-ocr

# Full run: ingest -> detect -> crop -> OCR, then auto-launch review UI
python main.py --input "Signature documents/some.pdf"
python main.py --input "Signature documents"          # folder of PDFs
python main.py --input <path> --no-review             # skip UI launch
python main.py --input <path> --dpi 300               # render DPI (default 300)

# Just the review UI (reads existing processed/ output)
python main.py --review
# equivalent: uvicorn review_app.server:app --host 127.0.0.1 --port 8000

# Push accepted/edited entries to Supabase (stub until creds set)
python main.py --push
```

No test suite or linter is configured yet.

## Architecture

The pipeline is intentionally split so each stage can be re-run independently against the on-disk artifacts of the previous one. **`processed/<pdf_stem>/` is the source of truth** — every stage reads from and writes to it, and the review UI is just a view over that folder.

### Per-PDF on-disk layout

```
processed/<pdf_stem>/
  pages/        page_0001.png ...        # rendered by ingest_pdf
  crops/        <entry_id>.png           # signature crops (overwritten on re-edit)
  entries/      <entry_id>.json          # one sidecar per detected signature
  manifest.json                          # summary listing, regenerated per run
```

Entry IDs are deterministic (`p{page:04d}_r{x}_{y}`) so re-running detection updates rather than duplicates.

### Stage flow (`main.py::process_pdf`)

1. **`src/ingest_pdf.py`** — PyMuPDF renders every page to PNG in parallel via `ProcessPoolExecutor`. This is the main lever for the <1 min target; do not serialize it.
2. **`src/detect_regions.py`** — per page, OpenCV Otsu + morphological close, then classify each contour as `image_region` vs `text_region` using an area-ratio + pixel-std-dev heuristic. Returns `Region` dataclasses in **page-pixel coordinates**.
3. **`src/crop_signature.py::detect_signature_in_region`** — currently a **local heuristic** (Canny + wide-kernel dilate, scoring wide-short contours biased to the bottom of the image region). The docstring flags this as the swap-in point for the Gemini Stage B call — replace the function body only; its signature (page path + `Region` + out path → bbox dict in page coordinates) is what the rest of the pipeline expects.
4. **`src/extract_text.py::ocr_region`** — Tesseract on the text region nearest each image region. `main._nearest_text` requires horizontal overlap and picks the closest one below.
5. **`src/store_locally.py`** — writes `entries/<id>.json` (defaults `review_status="pending"`, `pushed=False`, adds `created_at`) and regenerates `manifest.json`.

### Coordinate convention

All bboxes stored in JSON are **page-pixel coordinates at the render DPI** (default 300). The review UI's edit endpoint (`review_app/server.py::edit`) uses these same coordinates directly against `pages/page_XXXX.png` when re-cropping, so any new code producing bboxes must keep them in page space, not region-local space. `detect_signature_in_region` already handles the region→page offset before returning.

### Review UI (`review_app/`)

FastAPI + Jinja2 + vanilla JS. Mounts `processed/` at `/processed` so the browser loads page images and crops directly. Two mutation endpoints:

- `POST /entry/{pdf}/{id}/accept` — sets `review_status="accepted"`, optionally updates OCR fields. Does **not** touch the crop PNG.
- `POST /entry/{pdf}/{id}/edit` — takes a new `{x,y,w,h}` in page coordinates, re-crops `pages/page_XXXX.png` with Pillow, overwrites `crops/<id>.png`, updates `signature_bbox`, sets `review_status="edited"`.

The `/push` endpoint is a thin wrapper around `src.push_to_supabase.push_all`.

### Push (`src/push_to_supabase.py`)

Currently a **credential-gated stub**. When implementing: iterate `processed/*/entries/*.json` where `review_status in {accepted, edited}` and `pushed is False`; upload `crops/<id>.png` to Supabase Storage bucket `signatures`; insert a row; set `pushed=True` in the sidecar; log to `logs/push.log`. Push one at a time with a delay to avoid rate limits. `supabase` is commented out in `requirements.txt` — uncomment when wiring this up.

## Conventions

- Every module gets its logger via `src.logging_setup.get_logger(name)`; two names are in use — `pipeline` (everything under `src/` for processing) and `push` (Supabase). Logs go to `logs/<name>.log` and stderr. Do not `print()`.
- `processed/` and `logs/` are gitignored (working artifacts). Never commit them.
- Secrets live in `.env` (also gitignored); `.env.example` documents the expected keys (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `GEMINI_API_KEY`).
- The dataclass `src.detect_regions.Region` is the shared vocabulary between stages 2–4; extend it rather than passing raw tuples.
