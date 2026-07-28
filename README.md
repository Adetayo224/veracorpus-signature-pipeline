# VeraCorpus Signature Extraction & Review Pipeline

Local desktop pipeline that ingests multi-page PDF catalogues of sculpture entries, auto-detects and crops signature/inscription regions, OCRs the surrounding descriptive text, presents results in a local review UI for accept/re-crop, and on command pushes approved entries to Supabase.

**Performance target:** a 60-page PDF finishes local detection + cropping in well under one minute (excluding manual review and any API calls).

## Requirements

- Python 3.10+
- Tesseract OCR binary (`apt install tesseract-ocr`)
- Python deps in `requirements.txt`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in Supabase / API keys when needed
```

Source PDFs and guide videos are **not** stored in this repo — they live on Google Drive. Place PDFs to process into a local `Signature documents/` folder (gitignored).

## Usage

```bash
# Full run: ingest -> detect -> crop -> OCR, then auto-launch review UI
python main.py --input "Signature documents/some.pdf"
python main.py --input "Signature documents"          # folder of PDFs
python main.py --input <path> --no-review             # skip UI launch
python main.py --input <path> --dpi 300               # render DPI (default 300)

# Just the review UI (reads existing processed/ output)
python main.py --review

# Push accepted/edited entries to Supabase
python main.py --push
```

## On-disk layout

`processed/<pdf_stem>/` is the source of truth for every stage:

```
processed/<pdf_stem>/
  pages/        page_0001.png ...     # rendered pages
  crops/        <entry_id>.png        # signature crops
  entries/      <entry_id>.json       # one sidecar per detected signature
  manifest.json                       # summary listing
```

Entry IDs are deterministic (`p{page:04d}_r{x}_{y}`) so re-running detection updates rather than duplicates.

## Pipeline stages

1. **`src/ingest_pdf.py`** — PyMuPDF renders every page to PNG in parallel.
2. **`src/detect_regions.py`** — OpenCV Otsu + morphological close, classify contours as `image_region` vs `text_region`.
3. **`src/crop_signature.py`** — local Canny + dilate heuristic that scores wide-short contours toward the bottom of each image region.
4. **`src/extract_text.py`** — Tesseract OCR on the text region nearest each image region.
5. **`src/store_locally.py`** — writes `entries/<id>.json` and regenerates `manifest.json`.

All bboxes stored in JSON are in **page-pixel coordinates at the render DPI**.

## Review UI (`review_app/`)

FastAPI + Jinja2 + vanilla JS. Mounts `processed/` at `/processed`.

- `POST /entry/{pdf}/{id}/accept` — sets `review_status="accepted"`, optionally updates OCR fields.
- `POST /entry/{pdf}/{id}/edit` — takes a new `{x,y,w,h}` in page coordinates, re-crops from the page PNG, overwrites the crop, sets `review_status="edited"`.

## Push (`src/push_to_supabase.py`)

Credential-gated. Iterates `processed/*/entries/*.json` where `review_status in {accepted, edited}` and `pushed is False`, uploads the crop to the `signatures` Storage bucket, inserts a row, then flips `pushed=True`.

## Conventions

- Every module gets its logger via `src.logging_setup.get_logger(name)` (`pipeline` for processing, `push` for Supabase). No `print()`.
- `processed/` and `logs/` are gitignored working artifacts.
- Secrets live in `.env`; see `.env.example` for expected keys.
