from __future__ import annotations

# Text is now extracted directly from the PDF layer in `detect_regions`
# (see `_parse_entry_text`). Tesseract is intentionally NOT used, because
# the source PDFs already carry an extractable text layer whose accuracy
# and coordinates are far better than re-OCRing the scan. This module is
# kept as a stub so the older wiring in main.py and any external scripts
# don't break on import.


def ocr_region(*_args, **_kwargs) -> dict:
    return {
        "artist_name_claimed": "",
        "dates": "",
        "title": "",
        "raw_ocr_text": "",
    }
